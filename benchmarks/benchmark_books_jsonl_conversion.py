from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any


SOURCE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PARENT = SOURCE_ROOT.parent
if str(SOURCE_PARENT) not in sys.path:
    sys.path.insert(0, str(SOURCE_PARENT))

from rowpack.convert_jsonl import expand_jsonl_inputs
from rowpack.convert_jsonl_parallel import convert_jsonl_to_rowpack_parallel
from rowpack.native import load_native
from rowpack import RowPackReader


COLUMNS = [
    "variant",
    "status",
    "workers",
    "rows_per_block",
    "split_max_chars",
    "split_overlap_chars",
    "payload_format",
    "block_codec",
    "read_pattern",
    "read_workers",
    "read_block_size",
    "input_records",
    "input_mib",
    "rows_written",
    "blocks_written",
    "elapsed_s",
    "source_mib_per_s",
    "output_rows_per_s",
    "output_mib",
    "output_to_source_ratio",
    "compression_ratio_x",
    "space_savings_pct",
    "block_compression_ratio_x",
    "read_rows",
    "read_repeats",
    "read_elapsed_s",
    "read_rows_per_s",
    "read_stored_mib_per_s",
    "read_decoded_mib_per_s",
    "rowpack_path",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune RowPack JSONL conversion on book-style line-per-document corpora."
    )
    parser.add_argument("--input", default=str(SOURCE_ROOT / "examples" / "sample_books.jsonl"))
    parser.add_argument("--output-dir", default="results/books_jsonl_conversion_benchmark")
    parser.add_argument("--columns", nargs="+", default=["meta", "text"])
    parser.add_argument("--index-column", default="meta.short_book_title")
    parser.add_argument("--index-label-column", action="append", default=None)
    parser.add_argument("--split-column", action="append", default=["text"])
    parser.add_argument("--split-max-chars", nargs="+", type=int, default=[2048, 4096, 8192])
    parser.add_argument("--split-overlap-chars", type=int, default=128)
    parser.add_argument("--rows-per-block", nargs="+", type=int, default=[64, 128])
    parser.add_argument("--workers", nargs="+", type=int, default=default_workers())
    parser.add_argument("--payload-format", nargs="+", default=["json"], choices=["json", "cista"])
    parser.add_argument("--block-codec", nargs="+", default=["lzav_hi"], choices=["none", "lzav_default", "lzav_hi"])
    parser.add_argument("--native-module-dir", default=None)
    parser.add_argument("--max-input-lines", type=int, default=3)
    parser.add_argument("--max-input-bytes", type=int, default=None)
    parser.add_argument("--max-inflight-blocks", type=int, default=None)
    parser.add_argument("--progress-every-blocks", type=int, default=0)
    parser.add_argument("--read-pattern", default="random_block", choices=["sequential", "random_block"])
    parser.add_argument("--read-workers", nargs="+", type=int, default=[1])
    parser.add_argument("--read-block-size", type=int, default=128)
    parser.add_argument("--read-max-rows", type=int, default=0, help="Rows to read per repeat. 0 means all converted rows.")
    parser.add_argument("--read-repeats", type=int, default=3)
    parser.add_argument("--read-seed", type=int, default=1234)
    parser.add_argument("--no-read-benchmark", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-charts", action="store_true")
    return parser.parse_args()


def default_workers() -> list[int]:
    cpu_count = os.cpu_count() or 2
    values = [1, max(1, min(4, cpu_count - 1))]
    return sorted(set(values))


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    rowpack_dir = output_dir / "rowpacks"
    rowpack_dir.mkdir(parents=True, exist_ok=True)
    input_paths = expand_jsonl_inputs([args.input])
    index_label_columns = args.index_label_column or ["meta.url"]
    read_workers_values = unique_ints(args.read_workers)
    multiple_read_workers = len(read_workers_values) > 1

    if needs_native(args.payload_format, args.block_codec):
        load_native(args.native_module_dir)

    rows: list[dict[str, Any]] = []
    for payload_format in args.payload_format:
        for block_codec in args.block_codec:
            for workers in unique_ints(args.workers):
                for rows_per_block in unique_ints(args.rows_per_block):
                    for split_max_chars in unique_ints(args.split_max_chars):
                        variant = (
                            f"{payload_format}_{block_codec}_"
                            f"w{workers}_rpb{rows_per_block}_split{split_max_chars}"
                        )
                        output_path = rowpack_dir / f"{variant}.rowpack"
                        print(f"running {variant}", flush=True)
                        row = {
                            "variant": variant,
                            "status": "failed",
                            "workers": workers,
                            "rows_per_block": rows_per_block,
                            "split_max_chars": split_max_chars,
                            "split_overlap_chars": args.split_overlap_chars,
                            "payload_format": payload_format,
                            "block_codec": block_codec,
                            "read_pattern": args.read_pattern,
                            "read_block_size": args.read_block_size,
                            "rowpack_path": str(output_path),
                            "error": "",
                        }
                        try:
                            metrics = convert_jsonl_to_rowpack_parallel(
                                input_paths,
                                output=output_path,
                                columns=args.columns,
                                index_column=args.index_column,
                                index_label_columns=index_label_columns,
                                split_columns=set(args.split_column),
                                split_max_chars=split_max_chars,
                                split_overlap_chars=args.split_overlap_chars,
                                rows_per_block=rows_per_block,
                                payload_format=payload_format,
                                block_codec=block_codec,
                                native_module_dir=args.native_module_dir,
                                workers=workers,
                                max_inflight_blocks=args.max_inflight_blocks,
                                max_input_lines=args.max_input_lines,
                                max_input_bytes=args.max_input_bytes,
                                progress_every_blocks=args.progress_every_blocks,
                                overwrite=args.overwrite,
                            )
                            conversion_metrics = row_from_metrics(metrics)
                            if args.no_read_benchmark:
                                row.update(conversion_metrics)
                                row["read_workers"] = ""
                                row["status"] = "ready"
                                rows.append(row)
                            else:
                                for read_workers in read_workers_values:
                                    read_row = dict(row)
                                    if multiple_read_workers:
                                        read_row["variant"] = f"{variant}_rw{read_workers}"
                                    read_row.update(conversion_metrics)
                                    read_row["read_workers"] = read_workers
                                    read_metrics = benchmark_read_speed(
                                        output_path,
                                        native_module_dir=args.native_module_dir,
                                        read_pattern=args.read_pattern,
                                        read_workers=read_workers,
                                        read_block_size=args.read_block_size,
                                        read_max_rows=args.read_max_rows,
                                        read_repeats=args.read_repeats,
                                        read_seed=args.read_seed,
                                    )
                                    read_row.update(read_metrics)
                                    read_row["status"] = "ready"
                                    rows.append(read_row)
                        except Exception as exc:
                            row["error"] = f"{type(exc).__name__}: {exc}"
                            rows.append(row)
                            print(f"  failed: {row['error']}", flush=True)
                        write_outputs(rows, output_dir, no_charts=args.no_charts)

    write_outputs(rows, output_dir, no_charts=args.no_charts)
    print(f"wrote {output_dir / 'summary.csv'}")
    print(f"wrote {output_dir / 'summary.md'}")
    return 0


def needs_native(payload_formats: list[str], block_codecs: list[str]) -> bool:
    return any(payload == "cista" for payload in payload_formats) or any(codec != "none" for codec in block_codecs)


def unique_ints(values: list[int]) -> list[int]:
    return sorted(dict.fromkeys(int(value) for value in values))


def row_from_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    input_mib = metrics["input_bytes_read"] / (1024 * 1024)
    output_mib = metrics["output_bytes"] / (1024 * 1024)
    compression_ratio_x = metrics["input_bytes_read"] / metrics["output_bytes"] if metrics["output_bytes"] else None
    output_to_source_ratio = output_mib / input_mib if input_mib else None
    return {
        "input_records": metrics["input_records"],
        "input_mib": input_mib,
        "rows_written": metrics["rows_written"],
        "blocks_written": metrics["blocks_written"],
        "elapsed_s": metrics["elapsed_s"],
        "source_mib_per_s": metrics["source_mib_per_s"],
        "output_rows_per_s": metrics["output_rows_per_s"],
        "output_mib": output_mib,
        "output_to_source_ratio": output_to_source_ratio,
        "compression_ratio_x": compression_ratio_x,
        "space_savings_pct": (1.0 - output_to_source_ratio) * 100.0 if output_to_source_ratio is not None else None,
    }


def benchmark_read_speed(
    rowpack_path: Path,
    *,
    native_module_dir: str | None,
    read_pattern: str,
    read_workers: int,
    read_block_size: int,
    read_max_rows: int,
    read_repeats: int,
    read_seed: int,
) -> dict[str, Any]:
    with RowPackReader(rowpack_path, native_module_dir=native_module_dir) as reader:
        row_count = len(reader)
        metadata = reader.metadata
        stored_bytes_per_row = safe_div(float(metadata.get("block_payload_bytes") or 0), row_count)
        decoded_bytes_per_row = safe_div(float(metadata.get("block_uncompressed_bytes") or 0), row_count)
        block_compression_ratio_x = safe_div(
            float(metadata.get("block_uncompressed_bytes") or 0),
            float(metadata.get("block_payload_bytes") or 0),
        )

    rows_per_repeat = row_count if read_max_rows <= 0 else min(read_max_rows, row_count)
    read_workers = max(1, int(read_workers))
    read_repeats = max(1, int(read_repeats))

    total_rows = 0
    total_elapsed = 0.0
    if read_workers == 1:
        for repeat in range(read_repeats):
            elapsed, rows_read = _read_worker(
                str(rowpack_path),
                native_module_dir,
                read_pattern,
                rows_per_repeat,
                read_block_size,
                read_seed + repeat,
                0,
                1,
            )
            total_rows += rows_read
            total_elapsed += elapsed
    else:
        with ProcessPoolExecutor(max_workers=read_workers) as executor:
            for repeat in range(read_repeats):
                start = time.perf_counter()
                worker_args = [
                    (
                        str(rowpack_path),
                        native_module_dir,
                        read_pattern,
                        worker_row_count(rows_per_repeat, read_workers, worker_index),
                        read_block_size,
                        read_seed + repeat * read_workers + worker_index,
                        worker_index,
                        read_workers,
                    )
                    for worker_index in range(read_workers)
                ]
                results = list(executor.map(_read_worker_from_tuple, worker_args))
                elapsed = time.perf_counter() - start
                total_rows += sum(result[1] for result in results)
                total_elapsed += elapsed

    read_rows_per_s = total_rows / total_elapsed if total_elapsed else 0.0
    return {
        "block_compression_ratio_x": block_compression_ratio_x,
        "read_rows": total_rows,
        "read_repeats": read_repeats,
        "read_elapsed_s": total_elapsed,
        "read_rows_per_s": read_rows_per_s,
        "read_stored_mib_per_s": (stored_bytes_per_row * total_rows / (1024 * 1024)) / total_elapsed if total_elapsed else 0.0,
        "read_decoded_mib_per_s": (decoded_bytes_per_row * total_rows / (1024 * 1024)) / total_elapsed if total_elapsed else 0.0,
    }


def safe_div(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def worker_row_count(total_rows: int, workers: int, worker_index: int) -> int:
    base = total_rows // workers
    extra = 1 if worker_index < total_rows % workers else 0
    return base + extra


def _read_worker_from_tuple(args: tuple[str, str | None, str, int, int, int, int, int]) -> tuple[float, int]:
    return _read_worker(*args)


def _read_worker(
    rowpack_path: str,
    native_module_dir: str | None,
    read_pattern: str,
    target_rows: int,
    read_block_size: int,
    seed: int,
    worker_index: int,
    worker_count: int,
) -> tuple[float, int]:
    if target_rows <= 0:
        return 0.0, 0

    rows_read = 0
    with RowPackReader(rowpack_path, native_module_dir=native_module_dir) as reader:
        start = time.perf_counter()
        if read_pattern == "sequential":
            total_rows = len(reader)
            offset = (worker_index * target_rows) % total_rows if total_rows else 0
            for local_index in range(target_rows):
                reader.read_row((offset + local_index) % total_rows)
                rows_read += 1
        else:
            for _row in reader.iter_rows(
                read_pattern=read_pattern,
                max_rows=target_rows,
                read_block_size=read_block_size,
                seed=seed + worker_index * 1009 + worker_count * 9173,
            ):
                rows_read += 1
        elapsed = time.perf_counter() - start
    return elapsed, rows_read


def write_outputs(rows: list[dict[str, Any]], output_dir: Path, *, no_charts: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, output_dir / "summary.csv")
    write_markdown(rows, output_dir / "summary.md")
    if not no_charts:
        make_charts([row for row in rows if row.get("status") == "ready"], output_dir)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in COLUMNS})


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "| variant | status | conv workers | read workers | rows/block | split chars | codec | input MiB | output MiB | compression | write MiB/s | read rows/s | read decoded MiB/s | block compression | error |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {variant} | {status} | {workers} | {read_workers} | {rows_per_block} | {split_max_chars} | "
            "{block_codec} | {input_mib} | {output_mib} | {compression_ratio_x}x | {source_mib_per_s} | "
            "{read_rows_per_s} | {read_decoded_mib_per_s} | {block_compression_ratio_x}x | {error} |".format(
                variant=row.get("variant", ""),
                status=row.get("status", ""),
                workers=format_value(row.get("workers")),
                read_workers=format_value(row.get("read_workers")),
                rows_per_block=format_value(row.get("rows_per_block")),
                split_max_chars=format_value(row.get("split_max_chars")),
                block_codec=row.get("block_codec", ""),
                input_mib=format_value(row.get("input_mib"), digits=2),
                output_mib=format_value(row.get("output_mib"), digits=2),
                compression_ratio_x=format_value(row.get("compression_ratio_x"), digits=2),
                source_mib_per_s=format_value(row.get("source_mib_per_s"), digits=2),
                read_rows_per_s=format_value(row.get("read_rows_per_s"), digits=1),
                read_decoded_mib_per_s=format_value(row.get("read_decoded_mib_per_s"), digits=2),
                block_compression_ratio_x=format_value(row.get("block_compression_ratio_x"), digits=2),
                error=str(row.get("error") or "").replace("|", "\\|"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_value(value: Any, *, digits: int = 6) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def make_charts(rows: list[dict[str, Any]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    chart_dir = output_dir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    save_bar(rows, "source_mib_per_s", "JSONL Conversion Throughput", "MiB/s", chart_dir / "write_source_mib_per_s.png", "{:.2f}")
    save_bar(rows, "read_rows_per_s", "RowPack Read Throughput", "rows/s", chart_dir / "read_rows_per_s.png", "{:.1f}")
    save_bar(rows, "read_decoded_mib_per_s", "RowPack Read Throughput", "decoded MiB/s", chart_dir / "read_decoded_mib_per_s.png", "{:.2f}")
    save_bar(rows, "compression_ratio_x", "Source To RowPack Compression", "x smaller", chart_dir / "compression_ratio_x.png", "{:.2f}x")
    save_bar(rows, "output_mib", "RowPack Output Size", "MiB", chart_dir / "output_mib.png", "{:.2f}")


def save_bar(
    rows: list[dict[str, Any]],
    key: str,
    title: str,
    ylabel: str,
    path: Path,
    label_format: str,
) -> None:
    import matplotlib.pyplot as plt

    labels = [chart_label(row) for row in rows]
    values = [float(row.get(key) or 0.0) for row in rows]
    if not values:
        return
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.35), 5.2))
    bars = ax.bar(labels, values, color=[bar_color(str(row.get("block_codec", ""))) for row in rows])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=25)
    apply_zoomed_ylim(ax, values)
    label_bars(ax, bars, values, label_format)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def chart_label(row: dict[str, Any]) -> str:
    label = str(row["variant"])
    read_workers = row.get("read_workers")
    if read_workers not in (None, "") and f"_rw{read_workers}" not in label:
        label = f"{label}_rw{read_workers}"
    return label


def bar_color(block_codec: str) -> str:
    if block_codec == "lzav_hi":
        return "#f28e2b"
    if block_codec == "lzav_default":
        return "#59a14f"
    return "#376da8"


def apply_zoomed_ylim(ax, values: list[float]) -> None:
    min_value = min(values)
    max_value = max(values)
    if max_value <= 0:
        return
    margin = max((max_value - min_value) * 0.25, max_value * 0.03, 0.01)
    lower = max(0.0, min_value - margin)
    upper = max_value + margin
    if lower == 0 and min_value > 0:
        lower = min_value * 0.85
    if upper > lower:
        ax.set_ylim(lower, upper)


def label_bars(ax, bars, values: list[float], label_format: str) -> None:
    lower, upper = ax.get_ylim()
    span = max(upper - lower, 1e-9)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value - span * 0.04,
            label_format.format(value),
            ha="center",
            va="top",
            rotation=0,
            fontsize=8,
            color="white",
            fontweight="bold",
        )


if __name__ == "__main__":
    raise SystemExit(main())
