from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Let this file run from the examples directory before RowPack is installed as a
# normal Python package.
SOURCE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PARENT = SOURCE_ROOT.parent
if str(SOURCE_PARENT) not in sys.path:
    sys.path.insert(0, str(SOURCE_PARENT))

from rowpack import RowPackReader

from _summary import print_reader_summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Read a RowPack dataset and print a compact summary.")
    # Prefer the named --input form in docs because it reads naturally next to
    # the writer's --output flag. Keep the positional path as a convenience for
    # quick shell use and older README snippets.
    parser.add_argument("path", nargs="?", help="Input .rowpack path")
    parser.add_argument("--input", dest="input_path", help="Input .rowpack path")
    parser.add_argument("--native-module-dir", default=None, help="Directory containing rowpack_native")
    args = parser.parse_args()

    selected_path = args.input_path or args.path
    if selected_path is None:
        parser.error("provide an input RowPack path with --input or as a positional argument")

    path = Path(selected_path)
    # RowPackReader memory-maps the file and reads rows lazily. The context
    # manager closes the mmap/file handle even if summary printing fails.
    with RowPackReader(path, native_module_dir=args.native_module_dir) as reader:
        print_reader_summary(reader, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
