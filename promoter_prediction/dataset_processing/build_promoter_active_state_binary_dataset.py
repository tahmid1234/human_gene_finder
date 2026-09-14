"""


PROMOTER ACTIVE/INACTIVE CLASSIFICATION DATASET
binary dataset
completemd sequence
reverse mythylatiopn
- One sample per gene: DNA sequence in an asymmetric window around the
  TSS (mostly upstream, small downstream portion), no per-nt signal
  channel -- methylation is stored as ONE summary scalar per gene
  (mean methylation over the window), since the task predicts one
  label per gene, not one label per nucleotide.
- Label: 0=inactive (TPM==0), 1=active (TPM>=1), from GTEx
  Liver_Hepatocyte TPM. Genes with 0 < TPM < 1 are dropped (ambiguous).
- Window is NOT centered on TSS -- mostly upstream, small amount
  downstream, since the region after TSS is gene body, not promoter.

USAGE
    python 40_build_promoter_dataset.py \
        --fasta data/raw/genome.fa --gtf data/raw/annotation.gtf \
        --gtex-tpm data/raw/GTEx_..._gene_median_tpm.gct.gz \
        --methylation-bigwig data/raw/GSM5652233_....bigwig \
        --upstream 1000 --downstream 200 \
        --out data/processed/promoter_dataset_1200.h5

    # second window-size version, same command with different sizes:
    --upstream 1800 --downstream 200 --out data/processed/promoter_dataset_2000.h5
"""

import argparse
import gzip
import os

import gffutils
import h5py
import numpy as np
import pyBigWig
from pyfaidx import Fasta
from tqdm import tqdm

TEST_CHROMS = {"chr8", "chr9"}
VAL_CHROMS = {"chr21", "chr22"}

COMPLEMENT = str.maketrans("ACGTN", "TGCAN")

BASE_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}

INACTIVE_MAX_TPM = 0.0   # TPM <= this -> inactive
ACTIVE_MIN_TPM = 1.0     # TPM >= this -> active
                          # (0 < TPM < 1 -> dropped, ambiguous)


def one_hot_encode(seq):
    arr = np.zeros((len(seq), 5), dtype=np.float32)
    for i, base in enumerate(seq.upper()):
        arr[i, BASE_TO_IDX.get(base, 4)] = 1.0
    return arr


def build_label_db(gtf_path, db_path):
    if not os.path.exists(db_path):
        print("Building gffutils database from GTF (one-time)...")
        gffutils.create_db(
            gtf_path, dbfn=db_path, force=True, keep_order=True,
            merge_strategy="merge", sort_attribute_values=True,
            disable_infer_genes=True, disable_infer_transcripts=True,
        )
    return gffutils.FeatureDB(db_path)

def reverse_complement(seq):
    return seq.translate(COMPLEMENT)[::-1]

def strip_version(gene_id):
    return gene_id.split(".")[0]


def load_gtex_liver_hepatocyte_tpm(gtex_path):
    gene_tpm = {}
    with gzip.open(gtex_path, "rt") as f:
        f.readline()
        f.readline()
        header = f.readline().rstrip("\n").split("\t")
        col_idx = None
        for i, col in enumerate(header):
            if col.strip() == "Liver_Hepatocyte":
                col_idx = i
                break
        if col_idx is None:
            raise ValueError(f"'Liver_Hepatocyte' column not found. Columns: {header}")

        for line in f:
            fields = line.rstrip("\n").split("\t")
            gene_id = strip_version(fields[0])
            gene_tpm[gene_id] = float(fields[col_idx])

    print(f"Loaded Liver_Hepatocyte TPM for {len(gene_tpm)} genes.")
    return gene_tpm


def get_tss(gene):
    return gene.start if gene.strand == "+" else gene.end


def get_promoter_window(gene, rng, total_window=1200, min_upstream=900, max_upstream=1000):
    """Asymmetric window: mostly upstream of TSS, small amount downstream
    (downstream = into the gene body a little, since some core promoter
    elements sit just past the TSS -- see caveats in module docstring)."""
    upstream = rng.integers(min_upstream, max_upstream + 1)
    downstream = total_window - upstream
    tss = get_tss(gene)
    if gene.strand == "+":
        start = tss - upstream
        end = tss + downstream
    else:
        # "upstream" on - strand genes means toward HIGHER coordinates
        start = tss - downstream
        end = tss + upstream
    return start, end,upstream,downstream


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--gtf", required=True)
    ap.add_argument("--gtex-tpm", required=True)
    ap.add_argument("--methylation-bigwig", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--db-path", default="annotation.db")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--window", type=int, default=1200)
    ap.add_argument("--min_ups", type=int, default=900)
    ap.add_argument("--max_ups", type=int, default=1000)
    args = ap.parse_args()

    
    rng = np.random.default_rng(args.seed)
    fasta = Fasta(args.fasta)
    db = build_label_db(args.gtf, args.db_path)
    bw = pyBigWig.open(args.methylation_bigwig)
    gene_tpm = load_gtex_liver_hepatocyte_tpm(args.gtex_tpm)

    n_channels = 6  # DNA one-hot ONLY -- methylation is a separate scalar, not a channel
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    with h5py.File(args.out, "w") as h5:
        splits = {}
        for split_name in ("train", "val", "test"):
            grp = h5.create_group(split_name)
            splits[split_name] = {
                "x": grp.create_dataset("x", shape=(0, args.window, n_channels),
                                         maxshape=(None, args.window, n_channels),
                                         dtype="float32", chunks=(64, args.window, n_channels),
                                         compression="gzip"),
                "y": grp.create_dataset("y", shape=(0,), maxshape=(None,),
                                         dtype="uint8", chunks=(1024,), compression="gzip"),
                "methylation_mean": grp.create_dataset("methylation_mean", shape=(0,), maxshape=(None,),
                                                         dtype="float32", chunks=(1024,), compression="gzip"),
                "tpm": grp.create_dataset("tpm", shape=(0,), maxshape=(None,),
                                           dtype="float32", chunks=(1024,), compression="gzip"),
                "upstream_len": grp.create_dataset("upstream_len", shape=(0,), maxshape=(None,),
                                            dtype="int64", chunks=(1024,), compression="gzip"),

                "downstream_len": grp.create_dataset("downstream_len", shape=(0,), maxshape=(None,),
                                            dtype="int64", chunks=(1024,), compression="gzip"),
            }

        counts = {"train": [0, 0], "val": [0, 0], "test": [0, 0]}
        n_skipped_no_tpm = n_skipped_middle = n_skipped_edge = 0

        for gene in tqdm(db.features_of_type("gene"), desc="genes"):
            chrom = gene.seqid
            if chrom not in fasta:
                continue

            gene_id = strip_version(gene.id)
            if gene_id not in gene_tpm:
                n_skipped_no_tpm += 1
                continue
            tpm = gene_tpm[gene_id]

            if INACTIVE_MAX_TPM < tpm < ACTIVE_MIN_TPM:
                n_skipped_middle += 1
                continue
            label = 0 if tpm <= INACTIVE_MAX_TPM else 1

            start, end,upstream_len,downstream_len =  get_promoter_window(gene, rng,args.window,args.min_ups,args.max_ups)
            if start < 1 or end > len(fasta[chrom]):
                n_skipped_edge += 1
                continue

            seq = str(fasta[chrom][start - 1:end - 1])

            if gene.strand == "-":
                seq = reverse_complement(seq)
            if len(seq) != args.window or seq.upper().count("N") > 0.5 * args.window:
                continue

            dna = one_hot_encode(seq)  # (window, 5) -- DNA ONLY now

            meth_vals = bw.values(chrom, start - 1, end - 1, numpy=True)
            if gene.strand == "-":
                meth_vals = meth_vals[::-1]
            meth_vals = np.nan_to_num(meth_vals, nan=0.0).astype(np.float32)
            meth_vals = np.clip(meth_vals, 0.0, 1.0)
            x = np.concatenate([dna, meth_vals[:, None]], axis=-1)  # (window, 6)

            split_name = "test" if chrom in TEST_CHROMS else "val" if chrom in VAL_CHROMS else "train"
            ds = splits[split_name]
            n_old = ds["x"].shape[0]
            ds["x"].resize(n_old + 1, axis=0)
            ds["x"][n_old] = x
            ds["y"].resize(n_old + 1, axis=0)
            ds["y"][n_old] = label
            ds["tpm"].resize(n_old + 1, axis=0)
            ds["tpm"][n_old] = tpm
            ds["upstream_len"].resize(n_old + 1, axis=0)
            ds["upstream_len"][n_old] = upstream_len
            ds["downstream_len"].resize(n_old + 1, axis=0)
            ds["downstream_len"][n_old] = downstream_len
            counts[split_name][label] += 1

    for split_name, (n_inactive, n_active) in counts.items():
        print(f"{split_name}: {n_inactive} inactive, {n_active} active")
    print(f"skipped: {n_skipped_no_tpm} no GTEx match, {n_skipped_middle} ambiguous TPM, "
          f"{n_skipped_edge} too close to chrom edge")
    print(f"window = {args.window}nt ({args.upstream} upstream + {args.downstream} downstream of TSS)")


if __name__ == "__main__":
    main()