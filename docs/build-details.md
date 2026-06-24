# Build Details

The README keeps the first-run path short. This page collects the CMake details
that are useful when RowPack is embedded in another project or when CMake picks
a different Python than the one you intended.

## Parent Or Downstream Builds

From a parent checkout, point `-S` at the RowPack source directory and choose any
build directory you like:

```bash
cmake -S rowpack -B rowpack_build
cmake --build rowpack_build --config Release
```

Downstream CMake projects can also add RowPack directly:

```cmake
add_subdirectory(path/to/rowpack)
target_link_libraries(your_target PRIVATE rowpack_headers)
```

## Python Selection

RowPack uses CMake's normal `FindPython` flow and asks it to prefer the active
virtual environment or PATH Python. The native module is specific to the Python
version it was built against. If CMake chooses the wrong interpreter, configure
with the interpreter you want:

```bash
cmake -S . -B build -DPython_EXECUTABLE=$(command -v python3)
cmake --build build --config Release
```

## Native Module Discovery

Python APIs that need `rowpack_native` look in these places, in order:

- `--native-module-dir`, when the API or example accepts it
- `ROWPACK_NATIVE_DIR`
- the normal Python import path
- common local CMake output directories such as `build/`, `build/Release/`,
  `rowpack_build/`, and `rowpack_build/Release/`

For unusual build layouts, set the environment variable explicitly:

```bash
export ROWPACK_NATIVE_DIR=/path/to/rowpack/build
```
