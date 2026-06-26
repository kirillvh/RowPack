from __future__ import annotations

from pathlib import Path
from typing import Iterator

from .io import RowPackReader, TorchLikeRandom
from .native import load_native


class RowPackRows:
    """Iterable row source matching the benchmark's PyArrowParquetRows shape."""

    def __init__(
        self,
        paths: list[str],
        *,
        max_rows: int | None = None,
        read_pattern: str = "sequential",
        read_block_size: int = 32,
        seed: int = 0,
        native_module_dir: str | None = None,
    ):
        if not paths:
            raise ValueError("RowPackRows requires at least one .rowpack path")
        self.paths = [str(Path(path)) for path in paths]
        self.max_rows = max_rows
        self.read_pattern = read_pattern
        self.read_block_size = max(1, read_block_size)
        self.seed = seed
        self.native_module_dir = native_module_dir
        self.total_rows = self._count_rows()

    def __len__(self) -> int:
        if self.max_rows is None:
            return self.total_rows
        return self.max_rows

    def __iter__(self) -> Iterator[dict]:
        yielded = 0
        for file_idx, path in enumerate(self.paths):
            remaining = None if self.max_rows is None else self.max_rows - yielded
            if remaining is not None and remaining <= 0:
                return

            with RowPackReader(path, native_module_dir=self.native_module_dir) as reader:
                per_file_max = remaining
                if per_file_max is not None and self.read_pattern != "random_block":
                    per_file_max = min(per_file_max, len(reader))
                for row in reader.iter_rows(
                    read_pattern=self.read_pattern,
                    max_rows=per_file_max,
                    read_block_size=self.read_block_size,
                    seed=self.seed + file_idx,
                ):
                    yield row
                    yielded += 1
                    if self.max_rows is not None and yielded >= self.max_rows:
                        return

    def _count_rows(self) -> int:
        total = 0
        for path in self.paths:
            with RowPackReader(path, native_module_dir=self.native_module_dir) as reader:
                total += len(reader)
        return total


class NativeCistaVQARows:
    """Minimal native CISTA RowPack source for VQA training.

    Yields ``(row_id, text_pairs, image_bytes)`` where ``text_pairs`` is a list
    of ``(user, assistant)`` tuples and ``image_bytes`` is a list of byte
    payloads. This bypasses reconstruction of the original Hugging Face row.
    """

    def __init__(
        self,
        paths: list[str],
        *,
        max_rows: int | None = None,
        read_pattern: str = "sequential",
        read_block_size: int = 32,
        seed: int = 0,
        native_module_dir: str | None = None,
        native_decode_images: bool = True,
    ):
        if not paths:
            raise ValueError("NativeCistaVQARows requires at least one .rowpack path")
        self.paths = [str(Path(path)) for path in paths]
        self.max_rows = max_rows
        self.read_pattern = read_pattern
        self.read_block_size = max(1, read_block_size)
        self.seed = seed
        self.native = load_native(native_module_dir)
        self.native_decode_images = native_decode_images
        self.total_rows = self._count_rows()

    def __len__(self) -> int:
        if self.max_rows is None:
            return self.total_rows
        return self.max_rows

    def __iter__(self):
        yielded = 0
        for file_idx, path in enumerate(self.paths):
            remaining = None if self.max_rows is None else self.max_rows - yielded
            if remaining is not None and remaining <= 0:
                return

            reader = self.native.Reader(path)
            row_count = int(reader.row_count())
            if remaining is None:
                per_file_max = row_count
            elif self.read_pattern == "random_block":
                per_file_max = remaining
            else:
                per_file_max = min(remaining, row_count)
            for row_idx in self._iter_indices(row_count, per_file_max, self.seed + file_idx):
                yield reader.read_cista_vqa_row(row_idx, self.native_decode_images)
                yielded += 1
                if self.max_rows is not None and yielded >= self.max_rows:
                    return

    def _count_rows(self) -> int:
        total = 0
        for path in self.paths:
            total += int(self.native.Reader(path).row_count())
        return total

    def _iter_indices(self, total_rows: int, max_rows: int | None, seed: int):
        if self.read_pattern == "sequential":
            limit = min(max_rows if max_rows is not None else total_rows, total_rows)
            yield from range(limit)
            return

        if self.read_pattern != "random_block":
            raise ValueError(f"Unsupported RowPack read_pattern {self.read_pattern!r}")

        target_rows = max_rows if max_rows is not None else total_rows
        rng = TorchLikeRandom(seed)
        yielded = 0
        while yielded < target_rows:
            window_size = min(self.read_block_size, target_rows - yielded)
            max_start = max(0, total_rows - window_size)
            start = rng.randint_inclusive(max_start) if max_start else 0
            for index in range(start, start + window_size):
                yield index
                yielded += 1
                if yielded >= target_rows:
                    break
