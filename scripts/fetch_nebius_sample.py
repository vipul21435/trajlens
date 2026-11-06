"""Download a seeded random sample of the nebius/SWE-agent-trajectories dataset.

The dataset (CC BY 4.0, https://huggingface.co/datasets/nebius/SWE-agent-trajectories)
holds 80,036 SWE-agent runs with the final patch and whether the task's tests
passed on it (``target``). Rows are read through the public Hugging Face
datasets-server ``/rows`` endpoint, so no parquet shard is downloaded. Each
row is written as one JSON file that ``trajlens analyze --format chat`` reads;
the ``eval_logs`` column is dropped to keep the sample small.

    uv run python scripts/fetch_nebius_sample.py --out ../_data/trajlens/nebius

Standard library only.
"""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.parse
import urllib.request
from pathlib import Path

DATASET = "nebius/SWE-agent-trajectories"
ROWS_URL = "https://datasets-server.huggingface.co/rows"
LICENCE = "CC BY 4.0"
PAGE = "https://huggingface.co/datasets/nebius/SWE-agent-trajectories"


def fetch_rows(offset: int, length: int, retries: int = 8) -> dict:
    query = urllib.parse.urlencode(
        {"dataset": DATASET, "config": "default", "split": "train", "offset": offset,
         "length": length}
    )  # fmt: skip
    delay = 5.0
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(f"{ROWS_URL}?{query}", timeout=60) as resp:
                return json.load(resp)
        except OSError as exc:
            if attempt == retries - 1:
                raise
            wait = delay
            headers = getattr(exc, "headers", None)
            if headers is not None and str(headers.get("Retry-After", "")).isdigit():
                wait = max(wait, float(headers["Retry-After"]))
            time.sleep(min(wait, 120.0))
            delay *= 2
    raise RuntimeError("unreachable")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--blocks", type=int, default=100, help="random starting offsets")
    ap.add_argument("--block-size", type=int, default=4, help="consecutive rows per offset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pause", type=float, default=1.5, help="seconds between requests")
    args = ap.parse_args()

    first = fetch_rows(0, 1)
    total = int(first["num_rows_total"])
    rng = random.Random(args.seed)
    starts = sorted(rng.sample(range(0, total - args.block_size), args.blocks))
    args.out.mkdir(parents=True, exist_ok=True)
    written = []
    for start in starts:
        done = [args.out / f"row{start + k:05d}.json" for k in range(args.block_size)]
        if all(p.exists() for p in done):  # resume after an interrupted run
            written.extend(range(start, start + args.block_size))
            continue
        page = fetch_rows(start, args.block_size)
        for item in page["rows"]:
            row = dict(item["row"])
            row.pop("eval_logs", None)
            offset = int(item["row_idx"])
            row["run_id"] = f"row{offset:05d}"
            path = args.out / f"row{offset:05d}.json"
            path.write_text(json.dumps(row), encoding="utf-8")
            written.append(offset)
        time.sleep(args.pause)
    manifest = {
        "dataset": DATASET,
        "page": PAGE,
        "licence": LICENCE,
        "rows_total": total,
        "seed": args.seed,
        "blocks": args.blocks,
        "block_size": args.block_size,
        "row_offsets": written,
    }
    (args.out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {len(written)} rows of {total} to {args.out}")


if __name__ == "__main__":
    main()
