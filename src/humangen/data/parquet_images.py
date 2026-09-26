"""Streaming reader for Hugging Face image datasets stored as parquet shards."""

from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image


def list_shards(root: str | Path, pattern: str = "*.parquet") -> list[Path]:
    shards = sorted(Path(root).rglob(pattern))
    if not shards:
        raise FileNotFoundError(f"no parquet shards matching {pattern!r} under {root}")
    return shards


def count_rows(shards: list[Path]) -> int:
    return sum(pq.ParquetFile(s).metadata.num_rows for s in shards)


def decode_image(cell: dict, size: int | None = None) -> np.ndarray:
    """Decode an HF `Image` cell to uint8 HWC RGB, optionally resizing (Lanczos) to size x size."""
    img = Image.open(io.BytesIO(cell["bytes"])).convert("RGB")
    if size is not None and img.size != (size, size):
        img = img.resize((size, size), Image.LANCZOS)
    return np.asarray(img)


def iter_rows(shards: list[Path], columns: list[str], batch_rows: int = 256) -> Iterator[dict[str, list]]:
    """Yields column-dict batches, in shard order (deterministic global indexing)."""
    for shard in shards:
        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(batch_size=batch_rows, columns=columns):
            yield batch.to_pydict()
