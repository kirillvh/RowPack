from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

try:
    from torch.utils.data import IterableDataset, get_worker_info
except Exception:  # pragma: no cover - torch is optional for non-PyTorch users.
    IterableDataset = object  # type: ignore

    def get_worker_info():  # type: ignore
        return None

from .format import HEADER_SIZE, unpack_header
from .io import RowPackReader
from .native import load_native


ReadMode = Literal["sequential", "shuffle", "random"]
ReturnFormat = Literal["row", "native_vqa", "row_bytes"]


@dataclass
class RowPackLoaderState:
    """Reproducible block-stream cursor.

    In sequential mode, ``file_index`` and ``block_index`` are the actual next
    file/list-line and block to read. In shuffle/random mode, they are logical
    counters mixed with ``seed`` to produce deterministic pseudo-random file
    and block choices.
    """

    file_index: int = 0
    block_index: int = 0
    seed: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "file_index": int(self.file_index),
            "block_index": int(self.block_index),
            "seed": int(self.seed),
        }

    @classmethod
    def from_dict(cls, value: dict[str, int] | None) -> "RowPackLoaderState":
        if not value:
            return cls()
        return cls(
            file_index=int(value.get("file_index", 0)),
            block_index=int(value.get("block_index", 0)),
            seed=int(value.get("seed", 0)),
        )


@dataclass(frozen=True)
class RowPackFileInfo:
    path: Path
    row_count: int
    block_count: int
    payload_format: str
    blocks: tuple[tuple[int, int], ...]


class RowPackBlockDataset(IterableDataset):
    """PyTorch iterable dataset over one or more RowPack files.

    ``list_path`` is a plain text file with one ``.rowpack`` path per line.
    Relative paths are resolved from the list file's parent directory. Blank
    lines and ``#`` comments are ignored.

    The iterator schedules whole blocks, then yields rows sequentially inside
    each chosen block. That maps to random-window VLM training without paying
    fully random row-access costs.
    """

    def __init__(
        self,
        list_path: str | Path | None = None,
        *,
        paths: list[str | Path] | None = None,
        mode: ReadMode = "sequential",
        return_format: ReturnFormat = "row",
        state: RowPackLoaderState | dict[str, int] | None = None,
        seed: int = 0,
        max_rows: int | None = None,
        native_module_dir: str | None = None,
        native_decode_images: bool = True,
        shard_workers: bool = True,
    ):
        if list_path is None and not paths:
            raise ValueError("RowPackBlockDataset requires list_path or paths")
        if mode not in {"sequential", "shuffle", "random"}:
            raise ValueError("mode must be 'sequential', 'shuffle', or 'random'")
        if return_format not in {"row", "native_vqa", "row_bytes"}:
            raise ValueError("return_format must be 'row', 'native_vqa', or 'row_bytes'")

        self.paths = read_rowpack_list(list_path) if list_path is not None else [Path(path) for path in paths or []]
        if not self.paths:
            raise ValueError("RowPackBlockDataset path list is empty")

        self.mode = mode
        self.return_format = return_format
        self.state = RowPackLoaderState.from_dict(state if isinstance(state, dict) else None)
        if isinstance(state, RowPackLoaderState):
            self.state = RowPackLoaderState(state.file_index, state.block_index, state.seed)
        if seed != 0 or self.state.seed == 0:
            self.state.seed = int(seed)

        self.max_rows = max_rows
        self.native_module_dir = native_module_dir
        self.native_decode_images = native_decode_images
        self.shard_workers = shard_workers
        self.files = [read_file_info(path) for path in self.paths]
        self.total_rows = sum(info.row_count for info in self.files)
        self.total_blocks = sum(info.block_count for info in self.files)

        if self.return_format == "native_vqa":
            non_cista = [str(info.path) for info in self.files if info.payload_format != "cista"]
            if non_cista:
                raise ValueError("return_format='native_vqa' requires CISTA RowPack files: " + ", ".join(non_cista))

    def __len__(self) -> int:
        if self.max_rows is not None:
            return min(self.max_rows, self.total_rows)
        return self.total_rows

    def state_dict(self) -> dict[str, int]:
        return self.state.as_dict()

    def load_state_dict(self, value: dict[str, int]) -> None:
        self.state = RowPackLoaderState.from_dict(value)

    def __iter__(self) -> Iterator:
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info and self.shard_workers else 0
        worker_count = worker_info.num_workers if worker_info and self.shard_workers else 1
        target_rows = len(self)
        yielded = 0

        readers: dict[int, object] = {}
        try:
            for position, file_index, block_index in self._iter_block_schedule():
                if worker_count > 1 and position % worker_count != worker_id:
                    continue

                for row in self._iter_block_rows(file_index, block_index, readers):
                    if yielded >= target_rows:
                        return
                    yielded += 1
                    yield row
        finally:
            for reader in readers.values():
                close = getattr(reader, "close", None)
                if close is not None:
                    close()

    def _iter_block_schedule(self):
        if self.total_blocks == 0:
            return

        position = 0
        if self.mode == "sequential":
            file_index = self.state.file_index % len(self.files)
            block_index = self.state.block_index % max(self.files[file_index].block_count, 1)
            while True:
                info = self.files[file_index]
                if info.block_count:
                    yield position, file_index, block_index
                    position += 1
                file_index, block_index = self._next_sequential_block(file_index, block_index)
        else:
            file_counter = self.state.file_index
            block_counter = self.state.block_index
            while True:
                file_index = deterministic_index(self.state.seed, "file", file_counter, len(self.files))
                info = self.files[file_index]
                if info.block_count:
                    block_index = deterministic_index(self.state.seed, f"block:{file_index}", block_counter, info.block_count)
                    yield position, file_index, block_index
                    position += 1
                file_counter += 1
                block_counter += 1

    def _next_sequential_block(self, file_index: int, block_index: int) -> tuple[int, int]:
        info = self.files[file_index]
        block_index += 1
        if block_index < info.block_count:
            return file_index, block_index
        file_index = (file_index + 1) % len(self.files)
        return file_index, 0

    def _iter_block_rows(self, file_index: int, block_index: int, readers: dict[int, object]):
        info = self.files[file_index]
        if self.return_format == "native_vqa":
            reader = readers.get(file_index)
            if reader is None:
                reader = load_native(self.native_module_dir).Reader(info.path)
                readers[file_index] = reader

            start, row_count = info.blocks[block_index]
            stop = start + row_count
            for row_index in range(start, stop):
                try:
                    yield reader.read_cista_vqa_row(row_index, self.native_decode_images)
                except TypeError:
                    # Older rowpack_native builds always decode images and only
                    # accept the row index. Keep them usable for existing clones.
                    yield reader.read_cista_vqa_row(row_index)
            return

        reader = readers.get(file_index)
        if reader is None:
            reader = RowPackReader(info.path, native_module_dir=self.native_module_dir)
            readers[file_index] = reader

        start, row_count = info.blocks[block_index]
        stop = start + row_count
        if self.return_format == "row_bytes":
            native_reader = load_native(self.native_module_dir).Reader(info.path)
            for row_index in range(start, stop):
                yield bytes(native_reader.read_row_bytes(row_index))
            return

        for row_index in range(start, stop):
            yield reader.read_row(row_index)


def read_rowpack_list(list_path: str | Path) -> list[Path]:
    list_path = Path(list_path)
    if list_path.suffix == ".rowpack" and list_path.exists():
        return [list_path.resolve()]
    if not list_path.exists():
        parent = list_path.parent
        nearby = sorted(parent.glob("*.rowpack")) if parent.exists() else []
        if nearby:
            examples = "\n".join(f"  {path.name}" for path in nearby[:5])
            raise FileNotFoundError(
                f"RowPack list file not found: {list_path}\n\n"
                "A RowPack list is a plain UTF-8 text file with one .rowpack path per line. "
                "Create it with:\n\n"
                f"  python -m rowpack.make_list --input {parent} --output {list_path} --overwrite\n\n"
                f"Found {len(nearby)} .rowpack file(s) nearby, for example:\n{examples}"
            )
        raise FileNotFoundError(
            f"RowPack list file not found: {list_path}\n\n"
            "A RowPack list is a plain UTF-8 text file with one .rowpack path per line. "
            "If you have already converted data, create the list with:\n\n"
            f"  python -m rowpack.make_list --input path/to/rowpacks --output {list_path} --overwrite\n\n"
            "For the mm_infographic_vqa baseline, first create the RowPack file and list with:\n\n"
            "  python benchmarks/prepare_mm_infographic_vqa_rowpack.py "
            "--data-files data/variants/mm_infographic_vqa/uncompressed.parquet "
            "--output-dir data/variants/mm_infographic_vqa_rowpack "
            "--variant-name rowpack_cista_lzav_hi "
            "--payload-format cista --block-codec lzav_hi --rows-per-block 32 "
            "--rowpack-native-dir rowpack_build_py/Release --overwrite"
        )
    base = list_path.resolve().parent
    paths: list[Path] = []
    for raw_line in list_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip().lstrip("\ufeff")
        if not line or line.startswith("#"):
            continue
        path = Path(line)
        if not path.is_absolute():
            path = base / path
        paths.append(path)
    return paths


def read_file_info(path: Path) -> RowPackFileInfo:
    with path.open("rb") as handle:
        header = unpack_header(handle.read(HEADER_SIZE))
        handle.seek(header.metadata_offset)
        import json

        metadata = json.loads(handle.read(header.metadata_size).decode("utf-8"))
    with RowPackReader(path) as reader:
        blocks = tuple((int(block.start_row), int(block.row_count)) for block in reader.blocks)
    return RowPackFileInfo(
        path=path,
        row_count=int(header.row_count),
        block_count=int(header.block_count),
        payload_format=metadata.get("payload_format", "json"),
        blocks=blocks,
    )


def deterministic_index(seed: int, stream: str, counter: int, modulo: int) -> int:
    if modulo <= 0:
        raise ValueError("modulo must be > 0")
    payload = f"{int(seed)}:{stream}:{int(counter)}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") % modulo
