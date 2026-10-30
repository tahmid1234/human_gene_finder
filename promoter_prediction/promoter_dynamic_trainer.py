import argparse
import json
import math
import random
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

from promoter_multiscale_models.transformer_model import MultiModalTransformerRegressor
from promoter_multiscale_models.cross_attention_model import MultiModalCrossAttentionRegressor
from promoter_multiscale_models.multiscale_padding import (
    get_shared_mask_owners,
    pad_and_mask_sample,
    update_specs_for_padded_modalities,
)
import hashlib

EVAL_CROP_SEED = 42

HISTONE_NAMES = [
    "H3K27ac",
    "H3K27me3",
    "H3K36me3",
    "H3K4me1",
    "H3K4me3",
    "H3K9me3",
]

KNOWN_MODALITIES = ["dna", "meth"] + HISTONE_NAMES


def deterministic_upstream(gene_id, low, high, seed, window_name):
    key = f"{gene_id}|{seed}|{window_name}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(key).digest()[:8], "little")

    return low + (value % (high - low + 1))
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_channel_map(h5_path):
    """Resolve modality -> channel indices from H5 channel_names."""
    with h5py.File(h5_path, "r") as f:
        raw = f.attrs.get("channel_names")

    if raw is None:
        raise ValueError(
            "H5 file has no 'channel_names' attribute. "
            "The dataset builder must save channel_names."
        )

    if isinstance(raw, bytes):
        raw = raw.decode()

    names = [x.strip() for x in str(raw).split(",")]
    index = {name: i for i, name in enumerate(names)}

    required = ["A", "C", "G", "T", "N", "methylation", "cpg_observed"]
    for name in required:
        if name not in index:
            raise ValueError(f"Required channel {name!r} is missing")

    modality_channels = {
        "dna": [index[x] for x in ["A", "C", "G", "T", "N"]],
        "meth": [index["methylation"], index["cpg_observed"]],
    }

    for histone in HISTONE_NAMES:
        if histone in index:
            modality_channels[histone] = [index[histone]]

    return names, modality_channels


class MultiScalePromoterDataset(Dataset):
    """
    Read one full-length H5 file (normally 40 kb).

    Window rules:
      --short-modalities      -> 2 kb crop around TSS
      --semi-short-modalities -> 8 kb crop around TSS
      everything else        -> full stored sequence (normally 40 kb)

    For both 2 kb and 8 kb crops, the upstream length is sampled ONCE
    per sample in __init__ and is then fixed across every epoch.

    Defaults:
      2 kb: upstream 1000..1100, downstream 1000..900
      8 kb: upstream 4000..4100, downstream 4000..3900
    """

    def __init__(
        self,
        h5_path,
        split,
        modalities,
        modality_channels,
        short_modalities=None,
        semi_short_modalities=None,
        short_bp=2000,
        short_upstream_min=1000,
        short_upstream_max=1100,
        semi_short_bp=8000,
        semi_short_upstream_min=4000,
        semi_short_upstream_max=4100,
        crop_seed=42,
        pad_to_full=False,
    ):
        self.h5_path = h5_path
        self.split = split
        self.modalities = list(modalities)
        self.modality_channels = modality_channels

        self.short_modalities = set(short_modalities or [])
        self.semi_short_modalities = set(semi_short_modalities or [])

        self.short_bp = short_bp
        self.short_upstream_min = short_upstream_min
        self.short_upstream_max = short_upstream_max

        self.semi_short_bp = semi_short_bp
        self.semi_short_upstream_min = semi_short_upstream_min
        self.semi_short_upstream_max = semi_short_upstream_max

        self._file = None

        self.pad_to_full = pad_to_full
        self.mask_owners = get_shared_mask_owners(
            self.modalities, self.short_modalities, self.semi_short_modalities
            )

        with h5py.File(h5_path, "r") as f:
            if split not in f:
                raise ValueError(f"Split {split!r} not found in {h5_path}")
            
            raw_gene_ids = f[split]["gene_id"][:]
            self.gene_ids = [
                g.decode() if isinstance(g, bytes) else str(g)
                for g in raw_gene_ids
            ]

            self.length = len(f[split]["y"])
            self.full_bp = f[split]["x"].shape[1]

            if "tss_index" in f[split]:
                self.tss_key = "tss_index"
            elif "upstream_len" in f[split]:
                # After strand normalization, upstream_len is the TSS index.
                self.tss_key = "upstream_len"
            else:
                raise ValueError(
                    "Dataset needs either 'tss_index' or 'upstream_len'."
                )

        selected = set(self.modalities)

        for name in self.modalities:
            if name not in self.modality_channels:
                raise ValueError(
                    f"Modality {name!r} is unavailable in this H5 file."
                )

        bad_window_modalities = (
            self.short_modalities | self.semi_short_modalities
        ) - selected
        if bad_window_modalities:
            raise ValueError(
                "Every short/semi-short modality must also be present in "
                f"--modalities. Invalid: {sorted(bad_window_modalities)}"
            )

        overlap = self.short_modalities & self.semi_short_modalities
        if overlap:
            raise ValueError(
                "A modality cannot be both 2 kb and 8 kb. "
                f"Overlap: {sorted(overlap)}"
            )

        if short_upstream_min > short_upstream_max:
            raise ValueError(
                "short_upstream_min cannot exceed short_upstream_max"
            )
        if short_upstream_min < 0 or short_upstream_max > short_bp:
            raise ValueError(
                "2 kb upstream bounds must lie inside [0, short_bp]."
            )

        if semi_short_upstream_min > semi_short_upstream_max:
            raise ValueError(
                "semi_short_upstream_min cannot exceed "
                "semi_short_upstream_max"
            )
        if (
            semi_short_upstream_min < 0
            or semi_short_upstream_max > semi_short_bp
        ):
            raise ValueError(
                "8 kb upstream bounds must lie inside [0, semi_short_bp]."
            )

        # Split-specific deterministic RNG:
        # each sample gets a random crop position once, but that choice is
        # reproducible and does not change across epochs or DDP workers.
        split_offset = {
            "train": 0,
            "val": 1_000_000,
            "test": 2_000_000,
        }.get(split, 3_000_000)

        rng = np.random.default_rng(crop_seed + split_offset)
        self.fixed_short_upstream = np.array(
                [
                    deterministic_upstream(
                        gene_id,
                        short_upstream_min,
                        short_upstream_max,
                        EVAL_CROP_SEED,
                        "2kb",
                    )
                    for gene_id in self.gene_ids
                ],
                dtype=np.int32,
        )

        self.fixed_semi_short_upstream = np.array(
                [
                    deterministic_upstream(
                        gene_id,
                        semi_short_upstream_min,
                        semi_short_upstream_max,
                        EVAL_CROP_SEED,
                        "8kb",
                    )
                    for gene_id in self.gene_ids
                ],
                dtype=np.int32,
        )

        # if split in ("val", "test"):

        #     self.fixed_short_upstream = np.array(
        #         [
        #             deterministic_upstream(
        #                 gene_id,
        #                 short_upstream_min,
        #                 short_upstream_max,
        #                 EVAL_CROP_SEED,
        #                 "2kb",
        #             )
        #             for gene_id in self.gene_ids
        #         ],
        #         dtype=np.int32,
        #     )

        #     self.fixed_semi_short_upstream = np.array(
        #         [
        #             deterministic_upstream(
        #                 gene_id,
        #                 semi_short_upstream_min,
        #                 semi_short_upstream_max,
        #                 EVAL_CROP_SEED,
        #                 "8kb",
        #             )
        #             for gene_id in self.gene_ids
        #         ],
        #         dtype=np.int32,
        #     )

        # else:
        #     # Training can still depend on the experiment seed.
        #     rng = np.random.default_rng(crop_seed)

        #     self.fixed_short_upstream = rng.integers(
        #         short_upstream_min,
        #         short_upstream_max + 1,
        #         size=self.length,
        #         dtype=np.int32,
        #     )

        #     self.fixed_semi_short_upstream = rng.integers(
        #         semi_short_upstream_min,
        #         semi_short_upstream_max + 1,
        #         size=self.length,
        #         dtype=np.int32,
        #     )


    def _ensure_open(self):
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")

    def __len__(self):
        return self.length

    def _short_crop_bounds(self, tss, idx, full_len):
        upstream = int(self.fixed_short_upstream[idx])
        downstream = self.short_bp - upstream

        start = tss - upstream
        end = tss + downstream

        if start < 0 or end > full_len:
            raise ValueError(
                f"2 kb crop out of bounds for split={self.split}, idx={idx}: "
                f"tss={tss}, upstream={upstream}, downstream={downstream}, "
                f"start={start}, end={end}, full_len={full_len}"
            )

        return start, end

    def _semi_short_crop_bounds(self, tss, idx, full_len):
        upstream = int(self.fixed_semi_short_upstream[idx])
        downstream = self.semi_short_bp - upstream

        start = tss - upstream
        end = tss + downstream

        if start < 0 or end > full_len:
            raise ValueError(
                f"8 kb crop out of bounds for split={self.split}, idx={idx}: "
                f"tss={tss}, upstream={upstream}, downstream={downstream}, "
                f"start={start}, end={end}, full_len={full_len}"
            )

        return start, end

    def __getitem__(self, idx):
        self._ensure_open()
        grp = self._file[self.split]

        x = grp["x"][idx]
        y = float(grp["y"][idx])
        tss = int(grp[self.tss_key][idx])

        # Compute crop bounds only if that window size is actually used.
        short_bounds = None
        semi_short_bounds = None

        if self.short_modalities:
            short_bounds = self._short_crop_bounds(
                tss, idx, x.shape[0]
            )

        if self.semi_short_modalities:
            semi_short_bounds = self._semi_short_crop_bounds(
                tss, idx, x.shape[0]
            )

        modality_tensors = {}
        if self.pad_to_full:
            modality_tensors = pad_and_mask_sample(
                x_full=x,
                modalities=self.modalities,
                modality_channels=self.modality_channels,
                short_modalities=self.short_modalities,
                semi_short_modalities=self.semi_short_modalities,
                short_bounds=short_bounds,
                semi_short_bounds=semi_short_bounds,
                mask_owners=self.mask_owners,
            )
            return {
                "modalities": modality_tensors,
                "labels": torch.tensor(y, dtype=torch.float32),
            }

        for name in self.modalities:
            channels = self.modality_channels[name]

            if name in self.short_modalities:
                start, end = short_bounds
                arr = x[start:end, channels]

            elif name in self.semi_short_modalities:
                start, end = semi_short_bounds
                arr = x[start:end, channels]

            else:
                arr = x[:, channels]

            if arr.ndim == 1:
                arr = arr[:, None]

            modality_tensors[name] = torch.from_numpy(
                np.asarray(arr, dtype=np.float32)
            )

        return {
            "modalities": modality_tensors,
            "labels": torch.tensor(y, dtype=torch.float32),
        }


def collate_batch(batch):
    """HF Trainer collator for the nested modality dictionary."""
    names = batch[0]["modalities"].keys()

    modalities = {
        name: torch.stack(
            [item["modalities"][name] for item in batch],
            dim=0,
        )
        for name in names
    }

    labels = torch.stack(
        [item["labels"] for item in batch],
        dim=0,
    )

    return {
        "modalities": modalities,
        "labels": labels,
    }


def compute_metrics_regression(eval_pred):
    """Called by Hugging Face Trainer for validation and test evaluation."""
    predictions, labels = eval_pred

    predictions = np.asarray(predictions).reshape(-1)
    labels = np.asarray(labels).reshape(-1)

    mse = mean_squared_error(labels, predictions)

    try:
        pearson = float(pearsonr(labels, predictions).statistic)
    except Exception:
        pearson = float("nan")

    try:
        spearman = float(spearmanr(labels, predictions).statistic)
    except Exception:
        spearman = float("nan")

    return {
        "r2": float(r2_score(labels, predictions)),
        "mse": float(mse),
        "rmse": float(math.sqrt(mse)),
        "mae": float(mean_absolute_error(labels, predictions)),
        "pearson_r": pearson,
        "spearman_r": spearman,
        "pred_mean": float(predictions.mean()),
        "pred_std": float(predictions.std()),
        "pred_frac_0_0.1": float(
            ((predictions >= 0.0) & (predictions <= 0.1)).mean()
        ),
    }


def make_specs(
    dataset,
    modalities,
    short_modalities,
    semi_short_modalities,
):
    """
    Tell each modality-specific encoder the ACTUAL bp length it receives:
      short       -> 2 kb
      semi-short  -> 8 kb
      otherwise   -> full stored sequence (normally 40 kb)
    """
    short = set(short_modalities)
    semi = set(semi_short_modalities)

    specs = {}

    for name in modalities:
        if name in short:
            input_bp = dataset.short_bp
        elif name in semi:
            input_bp = dataset.semi_short_bp
        else:
            input_bp = dataset.full_bp

        specs[name] = {
            "in_channels": len(dataset.modality_channels[name]),
            "input_bp": input_bp,
        }

    return specs


class HFRegressionWrapper(nn.Module):
    """Adapter that makes the project backbones compatible with HF Trainer."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.loss_fn = nn.MSELoss()

    def forward(self, modalities=None, labels=None, **kwargs):
        logits = self.backbone(modalities)

        if logits.ndim == 1:
            logits = logits.unsqueeze(-1)

        output = {"logits": logits}

        if labels is not None:
            output["loss"] = self.loss_fn(
                logits.reshape(-1),
                labels.float().reshape(-1),
            )

        return output


def make_run_name(args):
    short_set = set(args.short_modalities)
    semi_set = set(args.semi_short_modalities)

    full_modalities = [
        m for m in args.modalities
        if m not in short_set and m not in semi_set
    ]

    mods = "-".join(full_modalities) if full_modalities else "none"
    short = (
        "-".join(args.short_modalities)
        if args.short_modalities
        else "none"
    )
    semi = (
        "-".join(args.semi_short_modalities)
        if args.semi_short_modalities
        else "none"
    )
    cross_mode = ""
    if args.route =='cross':
        cross_mode = args.cross_mode

    return (
        f"{args.route}"
        f"__mods_{mods}{cross_mode}"
        f"__short_{short}"
        f"__semi_{semi}"
        f"__tokens_{args.target_tokens}"
        f"_lr_{args.lr}"
        f"_dropout_{args.dropout}"
        f"_batch_{args.batch_size}"
        f"_dim_{args.branch_dim}"
    ).replace("/", "_")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data", required=True)
    ap.add_argument(
        "--route",
        choices=["transformer", "cross"],
        required=True,
    )

    ap.add_argument(
        "--modalities",
        nargs="+",
        required=True,
        help=(
            "Any subset of: dna meth H3K27ac H3K27me3 H3K36me3 "
            "H3K4me1 H3K4me3 H3K9me3"
        ),
    )

    ap.add_argument(
        "--short-modalities",
        nargs="*",
        default=[],
        help="Selected modalities that use the 2 kb TSS-centered crop.",
    )

    ap.add_argument(
        "--semi-short-modalities",
        nargs="*",
        default=[],
        help="Selected modalities that use the 8 kb TSS-centered crop.",
    )

    # 2 kb crop defaults:
    # 1000..1100 bp upstream and therefore 1000..900 bp downstream.
    ap.add_argument("--short-bp", type=int, default=2000)
    ap.add_argument("--short-upstream-min", type=int, default=1000)
    ap.add_argument("--short-upstream-max", type=int, default=1100)

    # 8 kb crop defaults:
    # 4000..4100 bp upstream and therefore 4000..3900 bp downstream.
    ap.add_argument("--semi-short-bp", type=int, default=8000)
    ap.add_argument("--semi-short-upstream-min", type=int, default=4000)
    ap.add_argument("--semi-short-upstream-max", type=int, default=4100)

    ap.add_argument(
        "--target-tokens",
        type=int,
        default=2000,
        help="Tokens per modality when entering the attention layers.",
    )

    ap.add_argument("--branch-dim", type=int, default=48)
    ap.add_argument("--transformer-dim", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument(
        "--cross-mode",
        choices=["hub", "full"],
        default="hub",
    )
    ap.add_argument("--hub-modality", default=None)

    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--pad-to-full",
        action="store_true",
        help="Pad 2kb/8kb modalities to 40kb with 0s and 2 shared presence masks.",
        )
    ap.add_argument(
        "--out-root",
        default=(
            "/mmfs1/projects/changhui.yan/mdtahmid.islam/"
            "Human_Genome/encoder_based_gtenefinding/"
            "human_gene_finder/promoter_prediction/model_dynamic"
        ),
    )

    args = ap.parse_args()
    set_seed(args.seed)

    all_requested = (
        args.modalities
        + args.short_modalities
        + args.semi_short_modalities
    )

    for name in all_requested:
        if name not in KNOWN_MODALITIES:
            raise ValueError(
                f"Unknown modality {name!r}. "
                f"Known modalities: {KNOWN_MODALITIES}"
            )

    if args.route == "cross" and len(args.modalities) < 2:
        raise ValueError(
            "Cross-attention requires at least two modalities."
        )

    channel_names, modality_channels = read_channel_map(args.data)

    dataset_common = dict(
        h5_path=args.data,
        modalities=args.modalities,
        modality_channels=modality_channels,

        short_modalities=args.short_modalities,
        semi_short_modalities=args.semi_short_modalities,

        short_bp=args.short_bp,
        short_upstream_min=args.short_upstream_min,
        short_upstream_max=args.short_upstream_max,

        semi_short_bp=args.semi_short_bp,
        semi_short_upstream_min=args.semi_short_upstream_min,
        semi_short_upstream_max=args.semi_short_upstream_max,

        crop_seed=args.seed,
    )

    train_ds = MultiScalePromoterDataset(
        split="train",
        **dataset_common,
    )
    val_ds = MultiScalePromoterDataset(
        split="val",
        **dataset_common,
    )
    test_ds = MultiScalePromoterDataset(
        split="test",
        **dataset_common,
    )

    specs = make_specs(
        train_ds,
        args.modalities,
        args.short_modalities,
        args.semi_short_modalities,
    )
    if args.pad_to_full:
        specs = update_specs_for_padded_modalities(
            specs,
            train_ds.full_bp,
            args.modalities,
            args.short_modalities,
            args.semi_short_modalities,
        )

    print("\nSelected modality input lengths:")
    for name, spec in specs.items():
        if name in args.short_modalities:
            window_name = "2 kb"
        elif name in args.semi_short_modalities:
            window_name = "8 kb"
        else:
            window_name = f"{spec['input_bp'] / 1000:g} kb"

        print(
            f"  {name:12s} -> "
            f"{spec['input_bp']:6d} bp ({window_name}), "
            f"{spec['in_channels']} channel(s)"
        )

    if args.route == "transformer":
        backbone = MultiModalTransformerRegressor(
            modality_specs=specs,
            target_tokens=args.target_tokens,
            branch_dim=args.branch_dim,
            transformer_dim=args.transformer_dim,
            transformer_layers=args.layers,
            transformer_heads=args.heads,
            dropout=args.dropout,
        )

    else:
        hub = args.hub_modality or args.modalities[0]

        if hub not in args.modalities:
            raise ValueError(
                f"Hub modality {hub!r} must be included in --modalities."
            )

        if args.branch_dim % args.heads != 0:
            raise ValueError(
                "--branch-dim must be divisible by --heads."
            )

        backbone = MultiModalCrossAttentionRegressor(
            modality_specs=specs,
            target_tokens=args.target_tokens,
            d_model=args.branch_dim,
            self_layers=max(1, args.layers // 2),
            self_heads=args.heads,
            cross_heads=args.heads,
            cross_mode=args.cross_mode,
            hub_modality=hub,
            dropout=args.dropout,
        )

    model = HFRegressionWrapper(backbone)

    run_dir = Path(args.out_root) / make_run_name(args)
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print("\nArguments:")
    print(args)
    print("\nH5 channels:", channel_names)
    print("Selected modality specs:", specs)
    print("Output directory:", run_dir)

    print(
        "\n2 kb crop: one fixed random crop per sample; "
        f"upstream range=[{args.short_upstream_min}, "
        f"{args.short_upstream_max}]."
    )
    print(
        "8 kb crop: one fixed random crop per sample; "
        f"upstream range=[{args.semi_short_upstream_min}, "
        f"{args.semi_short_upstream_max}]."
    )
    print("The window split is now deterministic across all training model")
    training_args = TrainingArguments(
        output_dir=str(run_dir),

        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,

        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        lr_scheduler_type="linear",
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=1.0,

        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",

        load_best_model_at_end=True,
        metric_for_best_model="r2",
        greater_is_better=True,

        save_total_limit=2,

        bf16=torch.cuda.is_available(),
        fp16=False,

        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=torch.cuda.is_available(),
        remove_unused_columns=False,

        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_batch,
        compute_metrics=compute_metrics_regression,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=10,
            )
        ],
    )

    trainer.train()

    best_checkpoint = trainer.state.best_model_checkpoint

    print("\n========================================")
    print("BEST VALIDATION CHECKPOINT")
    print(best_checkpoint)
    print("========================================")

    # load_best_model_at_end=True has already restored the best checkpoint.
    print("\nTEST evaluation is being run for checkpoint:")
    print(best_checkpoint)

    test_metrics = trainer.evaluate(
        eval_dataset=test_ds,
        metric_key_prefix="test",
    )

    print("\nTEST METRICS")
    for key, value in sorted(test_metrics.items()):
        if isinstance(value, (int, float)):
            print(f"{key:>24}: {value:.6f}")
        else:
            print(f"{key:>24}: {value}")

    with open(run_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    pred_output = trainer.predict(
        test_dataset=test_ds,
        metric_key_prefix="test_predict",
    )

    np.savez(
        run_dir / "test_predictions.npz",
        y_true=np.asarray(pred_output.label_ids).reshape(-1),
        y_pred=np.asarray(pred_output.predictions).reshape(-1),
    )

    print("\nFinished.")
    print("Best checkpoint used for test:", best_checkpoint)
    print("Run directory:", run_dir)


if __name__ == "__main__":
    main()
