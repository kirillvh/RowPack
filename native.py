from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import ModuleType


def load_native(native_module_dir: str | None = None) -> ModuleType:
    errors: list[str] = []
    for candidate in native_search_paths(native_module_dir):
        if candidate is None:
            try:
                return importlib.import_module("rowpack_native")
            except ModuleNotFoundError as exc:
                errors.append(str(exc))
                continue

        if candidate.exists():
            native_path = str(candidate.resolve())
            added = False
            if native_path not in sys.path:
                sys.path.insert(0, native_path)
                added = True
            try:
                return importlib.import_module("rowpack_native")
            except ModuleNotFoundError as exc:
                errors.append(f"{candidate}: {exc}")
                if added:
                    try:
                        sys.path.remove(native_path)
                    except ValueError:
                        pass

    message = "Could not import rowpack_native. Build RowPack first with `cmake -S . -B build && cmake --build build --config Release`."
    if errors:
        message += " Tried: " + "; ".join(errors)
    raise ModuleNotFoundError(message)


def native_search_paths(native_module_dir: str | None = None) -> list[Path | None]:
    paths: list[Path | None] = []
    if native_module_dir:
        paths.append(Path(native_module_dir))

    env_path = os.environ.get("ROWPACK_NATIVE_DIR")
    if env_path:
        paths.append(Path(env_path))

    # Try the regular import path first, then common local CMake output dirs.
    paths.append(None)

    cwd = Path.cwd()
    module_path = Path(__file__).resolve()
    package_dir = module_path.parent
    package_root = package_dir.parent if package_dir.name == "rowpack" else package_dir
    for root in dict.fromkeys([cwd, package_dir, package_root, *cwd.parents]):
        paths.extend(
            [
                root / "rowpack_build" / "Release",
                root / "rowpack_build",
                root / "build" / "Release",
                root / "build",
            ]
        )

    deduped: list[Path | None] = []
    seen = set()
    for path in paths:
        key = None if path is None else str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped
