import os
import math
import argparse
import numpy as np
import torch
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pos", required=True, help="POS, e.g. NOUN")
    parser.add_argument("--embeddings", default=None)
    parser.add_argument("--block-size", type=int, default=4096)
    parser.add_argument("--output-dir", default="../data/similarities/{pos}")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.embeddings is None:
        args.embeddings = f"../data/embedding/{args.pos}_embeddings.npy"

    output_dir = args.output_dir.format(pos=args.pos)
    os.makedirs(output_dir, exist_ok=True)

    rank = int(os.environ.get("SLURM_PROCID", 0))
    n_tasks = int(os.environ.get("SLURM_NTASKS", 1))
    gpus_per_node = int(os.environ.get("SLURM_GPUS_ON_NODE", 1))

    gpu_id = rank % gpus_per_node
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    print(f"[Rank {rank}] Using device: {device}", flush=True)

    emb = np.load(args.embeddings, mmap_mode="r")
    n, d = emb.shape
    print(f"[Rank {rank}] Embeddings: {n} x {d}", flush=True)

    block_size = args.block_size
    n_blocks = math.ceil(n / block_size)

    block_pairs = [
        (i, j)
        for idx, (i, j) in enumerate(
            ( (i, j) for i in range(n_blocks) for j in range(i, n_blocks) )
        )
        if idx % n_tasks == rank
    ]

    print(f"[Rank {rank}] Assigned block pairs: {len(block_pairs)}", flush=True)

    pbar = tqdm(block_pairs, desc=f"Rank {rank}", position=rank)

    for bi, bj in pbar:
        i0 = bi * block_size
        i1 = min((bi + 1) * block_size, n)
        j0 = bj * block_size
        j1 = min((bj + 1) * block_size, n)

        A = torch.from_numpy(emb[i0:i1]).to(device).float()
        B = torch.from_numpy(emb[j0:j1]).to(device).float()

        # Normalize
        A = torch.nn.functional.normalize(A, p=2, dim=1)
        B = torch.nn.functional.normalize(B, p=2, dim=1)

        sims = A @ B.T
        sims_cpu = sims.cpu().numpy()

        tag = f"bi{bi}_bj{bj}_rank{rank}"

        np.save(
            os.path.join(output_dir, f"sims_{tag}.npy"),
            sims_cpu.astype(np.float32)
        )

        with open(os.path.join(output_dir, f"meta_{tag}.txt"), "w") as f:
            f.write(f"{i0}\t{i1}\t{j0}\t{j1}\n")

        del A, B, sims

    print(f"[Rank {rank}] Done", flush=True)


if __name__ == "__main__":
    main()