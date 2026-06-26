from __future__ import annotations

import argparse
import glob
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create a RowPack list file for PyTorch/nanoVLM loaders.")
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="A .rowpack file, directory, or glob. Repeat for multiple locations.",
    )
    parser.add_argument("--output", default="rowpacks.txt", help="Output list file path.")
    parser.add_argument(
        "--absolute",
        action="store_true",
        help="Write absolute paths. By default, paths are relative to the list file directory when possible.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    paths = expand_rowpack_inputs(args.input)
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} already exists; pass --overwrite")
    write_rowpack_list(paths, output, absolute=args.absolute)
    print(f"wrote {len(paths)} RowPack path(s) to {output}")
    return 0


def expand_rowpack_inputs(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        candidate = Path(item)
        if candidate.is_dir():
            paths.extend(sorted(candidate.glob("*.rowpack")))
            continue
        matches = [Path(match) for match in sorted(glob.glob(item))] if has_glob_magic(item) else []
        if matches:
            paths.extend(path for path in matches if path.is_file() and path.suffix == ".rowpack")
            continue
        if candidate.exists() and candidate.is_file() and candidate.suffix == ".rowpack":
            paths.append(candidate)
            continue
        raise FileNotFoundError(f"No .rowpack files matched {item!r}")

    deduped = list(dict.fromkeys(path.resolve() for path in paths))
    if not deduped:
        raise FileNotFoundError("No .rowpack files matched")
    return deduped


def write_rowpack_list(paths: list[Path], output: str | Path, *, absolute: bool = False) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    base = output.resolve().parent
    lines = []
    for path in paths:
        resolved = Path(path).resolve()
        if absolute:
            lines.append(str(resolved))
        else:
            lines.append(str(relative_to_base(resolved, base)))
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def relative_to_base(path: Path, base: Path) -> Path:
    try:
        return path.relative_to(base)
    except ValueError:
        return path


def has_glob_magic(value: str) -> bool:
    return any(char in value for char in "*?[")


if __name__ == "__main__":
    raise SystemExit(main())
