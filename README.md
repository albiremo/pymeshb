# pymeshb

Python `ctypes` wrapper around [libMeshb](https://github.com/LoicMarechal/libMeshb),
the reference library for the Gamma Mesh Format (GMF). Reads `.mesh` (ASCII) and
`.meshb` (binary) files into plain Python dictionaries of NumPy arrays.

## Requirements

- Python >= 3.8
- [NumPy](https://numpy.org/)
- **libmeshb8** compiled and installed as a shared library — see
  [LoicMarechal/libMeshb](https://github.com/LoicMarechal/libMeshb) for build
  instructions. `pymeshb` loads it at runtime via `ctypes`; it is not bundled.

## Installation

Clone this repository and install it with pip:

```bash
git clone <this-repo-url>
cd pymeshb
pip install .
```

Or, for an editable/development install:

```bash
pip install -e .
```

### Locating libmeshb8

At import time, `pymeshb` searches for the shared library in this order:

1. `ctypes.util.find_library` (standard system search paths)
2. Common install directories (`/usr/local/lib`, `/usr/lib`, `/opt/homebrew/lib`, …
   depending on OS)
3. The `LIBMESHB_PATH` environment variable, pointing directly at the `.so` /
   `.dylib` / `.dll` file

If none of these find it, set the environment variable before running your script:

```bash
export LIBMESHB_PATH=/path/to/libmeshb8.so   # Linux
export LIBMESHB_PATH=/path/to/libmeshb8.dylib  # macOS
```

or add its directory to `LD_LIBRARY_PATH` (Linux) / `DYLD_LIBRARY_PATH` (macOS).

## Quick start

```python
from pymeshb import read_mesh

mesh = read_mesh("cube.meshb")

print(mesh["Dimension"])           # 3
print(mesh["Vertices"]["xyz"])     # (N, 3) float64 – coordinates
print(mesh["Vertices"]["ref"])     # (N,)   int32   – reference tags
print(mesh["Tetrahedra"]["nodes"]) # (M, 4) int32   – 1-based vertex indices
print(mesh["Tetrahedra"]["ref"])   # (M,)   int32
```

### Command line

```bash
python -m pymeshb cube.meshb [Vertices Triangles ...]
```

Prints the keywords present in the file and a summary of the arrays read.

## Output dictionary layout

Every keyword present in the file becomes a key (e.g. `"Vertices"`,
`"Triangles"`). The value type depends on the keyword family:

- **Mesh-entity keywords** (`Vertices`, `Edges`, `Triangles`, …):
  - `{"xyz": float64 (N, dim)}` — `Vertices` only
  - `{"nodes": int32 (N, ncol)}` — connectivity (1-based indices)
  - `{"ref": int32 (N,)}` — reference tag
  - Pure-index keywords (`Corners`, `Ridges`, …) return just an
    `int32 (N,)` array.
- **Solution keywords** (`SolAtVertices`, `SolAtTetrahedra`, …):
  - `{"data": float64 (N, sol_size), "sol_types": [int, ...]}`
    (`GmfSca=1`, `GmfVec=2`, `GmfSymMat=3`, `GmfMat=4`)
- **Scalar / header keywords**:
  - `"Dimension"`, `"Version"`, `"Iterations"` → `int`
  - `"Time"` → `float`
  - `"BoundingBox"` → `{"lo": (dim,), "hi": (dim,)}` float64 arrays

Node indices are 1-based, as stored in the file — subtract 1 for 0-based
Python indexing. Only keywords actually present in the file appear in the
output dict.

## API

| Function | Description |
| --- | --- |
| `read_mesh(filepath, keywords=None, verbose=False)` | Read a mesh/solution file into a dict. |
| `list_keywords(filepath)` | List keywords present in a file without reading their data. |
| `mesh_summary(mesh)` | Human-readable summary string of a mesh dict. |
| `split_sol(sol, dim, names=None)` | Split a packed `SolAt*` array into named fields. |
| `sol_field_sizes(sol_types, dim)` | Number of components per solution field. |

## License

MIT — see [LICENSE](LICENSE).
