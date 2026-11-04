"""Encode dense embeddings into the SAE latent space, appending a ``"sae embeddings"`` column.

Reads the SAE architecture and ``--normalize`` flag from the experiment folder's sae_config.json.
Accepts a single pkl or a directory of parts; writes ``<name>_with_sae`` pickles."""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.sae_architectures import load_sae_checkpoint
from src.utils.datasets import EmbDataset


def compute_activations(
    df: pd.DataFrame,
    model: nn.Module,
    device: torch.device,
    batch: int,
    num_workers: int = 2,
    sparse: bool = False,
) -> pd.DataFrame:
    """Encode every row in batches; return a DataFrame with 'sae embeddings' + 'reconstructed embeddings'."""
    X = np.vstack(df["embeddings"].apply(lambda v: np.asarray(v, dtype=np.float32)).values)
    positions = np.arange(len(df), dtype=np.int64)
    loader = DataLoader(EmbDataset(X, positions),
                        batch_size=batch, shuffle=False, num_workers=num_workers)
    sae_records: dict = {}
    rec_records: dict = {}
    model.eval()
    with torch.no_grad():
        for xb, pos in loader:
            xb = xb.view(xb.size(0), -1).to(torch.float32).to(device)
            _m = model.module if hasattr(model, "module") else model
            z = _m.encode(xb)
            x_hat = _m.decode(z)
            z_np = z.cpu().numpy()
            x_hat_np = x_hat.cpu().numpy()
            for i, p in enumerate(pos.tolist()):
                idx = df.index[int(p)]
                if sparse:
                    nz = np.flatnonzero(z_np[i])
                    sae_records[idx] = [(nz.astype(np.int32), z_np[i][nz].astype(np.float32))]
                else:
                    sae_records[idx] = [z_np[i]]
                rec_records[idx] = [x_hat_np[i]]
    out = pd.DataFrame.from_dict(sae_records, orient="index", columns=["sae embeddings"])
    out["reconstructed embeddings"] = pd.DataFrame.from_dict(
        rec_records, orient="index", columns=["reconstructed embeddings"]
    )["reconstructed embeddings"]
    return out


def convert_file(emb_file, model, device, batch, output_dir, hidden, sparse=False):
    """Read one pkl, encode through the SAE, save '<name>_with_sae' alongside in ``output_dir``. The
    checkpoint's latent width is recorded in the output's ``df.attrs``."""
    df = pd.read_pickle(emb_file)
    enc = compute_activations(df, model, device, batch, sparse=sparse)
    if isinstance(df.index[0], int) and isinstance(enc.index[0], str):
        df.index = df.index.astype(str)
    df.loc[enc.index, "sae embeddings"] = enc["sae embeddings"]
    df.loc[enc.index, "reconstructed embeddings"] = enc["reconstructed embeddings"]
    base, ext = os.path.splitext(os.path.basename(emb_file))
    out_path = os.path.join(output_dir, base + "_with_sae" + ext)
    df.attrs["sae_hidden"] = int(hidden)
    df.to_pickle(out_path)
    print(f"Saved {out_path}")


def convert_original_dimensions(emb_file, output_dir):
    """Save '<name>_with_sae' with the embeddings copied into 'sae embeddings': each original
    dimension is a unit."""
    df = pd.read_pickle(emb_file)
    df["sae embeddings"] = df["embeddings"]
    base, ext = os.path.splitext(os.path.basename(emb_file))
    out_path = os.path.join(output_dir, base + "_with_sae" + ext)
    df.to_pickle(out_path)
    print(f"Saved {out_path}")


def main():
    """CLI: load checkpoint + sae_config, encode one pkl or a directory of part files."""
    parser = argparse.ArgumentParser(description="Encode embeddings into SAE latent space")
    parser.add_argument("--emb-file", type=str, required=True,
                        help="Pickle file or directory of part files")
    parser.add_argument("--checkpoint", type=str, default="best_sae.pth")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--experiment-folder", type=str, default="./")
    parser.add_argument("--output-subdir", type=str, default=None,
                        help="Output subdirectory under experiment-folder (default 'converted')")
    parser.add_argument("--sparse-sae", action="store_true", default=False,
                        help="Force sparse (indices, values) storage even for non-TopK architectures")
    parser.add_argument("--dense-sae", action="store_true", default=False,
                        help="Force dense storage even for TopK architectures (default: TopK is sparse)")
    parser.add_argument("--use-original-dimensions", action="store_true",
                        help="No checkpoint: copy the embeddings into 'sae embeddings', so each "
                             "original dimension is a unit")
    args = parser.parse_args()

    if os.path.isdir(args.emb_file):
        emb_files = sorted(
            p for p in (os.path.join(args.emb_file, f) for f in os.listdir(args.emb_file))
            if os.path.isfile(p) and "_with_sae" not in os.path.basename(p)
            and not os.path.basename(p).endswith(".json")
        )
        if not emb_files:
            raise FileNotFoundError(f"No embedding files found in {args.emb_file}")
        print(f"Found {len(emb_files)} parts to convert")
    else:
        emb_files = [args.emb_file]

    subdir = args.output_subdir or "converted"
    output_dir = os.path.join(args.experiment_folder, subdir,
                              "corpus" if os.path.isdir(args.emb_file) else "")
    os.makedirs(output_dir, exist_ok=True)

    if args.use_original_dimensions:
        for emb_file in emb_files:
            convert_original_dimensions(emb_file, output_dir)
        return

    config_path = os.path.join(args.experiment_folder, "sae_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"sae_config.json not found in {args.experiment_folder}")
    with open(config_path) as f:
        cfg = json.load(f)
    print(f"Loaded sae_config.json from {args.experiment_folder}")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    input_dim = cfg.get("input_dim")
    if not input_dim:
        sample_df = pd.read_pickle(emb_files[0])
        input_dim = len(np.array(sample_df["embeddings"].iloc[0], dtype=np.float32))
        del sample_df

    model = load_sae_checkpoint(args.experiment_folder, args.checkpoint, device, input_dim=input_dim)

    is_topk = hasattr(model, "k")
    sparse = args.sparse_sae or (is_topk and not args.dense_sae)
    if sparse:
        print("Storing SAE embeddings in sparse (indices, values) format"
              + (" (default for TopK)" if is_topk and not args.sparse_sae else " (--sparse-sae)"))
    for emb_file in emb_files:
        print(f"Converting {emb_file}")
        convert_file(emb_file, model, device, args.batch, output_dir, cfg["hidden"], sparse=sparse)


if __name__ == "__main__":
    main()
