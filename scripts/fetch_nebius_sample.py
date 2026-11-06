"""Download a seeded random sample of a nebius trajectory dataset.

Two datasets are known, both CC BY 4.0:

``swe-agent``  nebius/SWE-agent-trajectories, 80,036 SWE-agent runs with the
               final patch and whether the task's tests passed on it
               (``target``); read with ``trajlens analyze --format chat``.
``openhands``  nebius/SWE-rebench-openhands-trajectories, 67,074 OpenHands
               function-calling runs with ``model_patch`` and ``resolved``;
               read with ``trajlens analyze --format toolcalls``.

Rows are read through the public Hugging Face datasets-server ``/rows``
endpoint, so no parquet shard is downloaded. Each row is written as one JSON
file; columns trajlens does not read (test logs, tool schemas) are dropped to
keep the sample small.

    uv run python scripts/fetch_nebius_sample.py --out nebius-sample
    uv run python scripts/fetch_nebius_sample.py --dataset openhands --blocks 20 --out oh-sample

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

ROWS_URL = "https://datasets-server.huggingface.co/rows"
LICENCE = "CC BY 4.0"
DATASETS = {
    "swe-agent": ("nebius/SWE-agent-trajectories", ("eval_logs",)),
    "openhands": ("nebius/SWE-rebench-openhands-trajectories", ("tools",)),
}


def fetch_rows(dataset: str, offset: int, length: int, retries: int = 8) -> dict:
    query = urllib.parse.urlencode(
        {"dataset": dataset, "config": "default", "split": "train", "offset": offset,
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
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="swe-agent")
    ap.add_argument("--blocks", type=int, default=100, help="random starting offsets")
    ap.add_argument("--block-size", type=int, default=4, help="consecutive rows per offset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pause", type=float, default=1.5, help="seconds between requests")
    args = ap.parse_args()

    dataset, drop = DATASETS[args.dataset]
    first = fetch_rows(dataset, 0, 1)
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
        page = fetch_rows(dataset, start, args.block_size)
        for item in page["rows"]:
            row = dict(item["row"])
            for col in drop:
                row.pop(col, None)
            offset = int(item["row_idx"])
            row["run_id"] = f"row{offset:05d}"
            path = args.out / f"row{offset:05d}.json"
            path.write_text(json.dumps(row), encoding="utf-8")
            written.append(offset)
        time.sleep(args.pause)
    manifest = {
        "dataset": dataset,
        "page": f"https://huggingface.co/datasets/{dataset}",
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
