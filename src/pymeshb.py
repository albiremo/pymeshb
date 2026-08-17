"""
pymeshb.py — Python ctypes wrapper for libMeshb 8.x (GMF format reader)
===========================================================================

Reads .mesh (ASCII) or .meshb (binary) files produced by the Gamma Mesh Format
library into a plain Python dict keyed by GMF keyword names.

Dependencies
------------
- libmeshb8 compiled and installed  →  https://github.com/LoicMarechal/libMeshb
- numpy

Quick-start
-----------
    from pymeshb import read_mesh

    mesh = read_mesh("cube.meshb")

    print(mesh["Dimension"])          # 3
    print(mesh["Vertices"]["xyz"])    # (N, 3) float64 – coordinates
    print(mesh["Vertices"]["ref"])    # (N,)   int32  – reference tags
    print(mesh["Tetrahedra"]["nodes"])# (M, 4) int32  – 1-based vertex indices
    print(mesh["Tetrahedra"]["ref"])  # (M,)   int32

Output dictionary layout
------------------------
Every present keyword becomes a key (string, e.g. "Vertices", "Triangles").
The value type depends on the keyword family:

  Mesh-entity keywords (Vertices, Edges, Triangles, …):
    {"xyz":   float64 (N, dim)  }   ← for Vertices only
    {"nodes": int32   (N, ncol) }   ← connectivity (1-based)
    {"ref":   int32   (N,)      }   ← reference tag (always last column)
    Single-column index keywords (Corners, Ridges, …) return just the
    int32 array of shape (N,).

  Solution keywords (SolAtVertices, SolAtTetrahedra, …):
    {"data":     float64 (N, sol_size) }
    {"sol_types": list of ints          }   ← GmfSca=1, GmfVec=2, …

  Scalar / header keywords:
    "Dimension"  → int
    "Version"    → int
    "Iterations" → int
    "Time"       → float
    "BoundingBox"→ (lo, hi) each a (dim,) float64 array

Notes
-----
- Node indices are 1-based (as stored in the file); subtract 1 for 0-based Python.
- This wrapper uses GmfGetBlock for bulk reads: one C call per keyword,
  not one call per element, so it handles large meshes efficiently.
- SolAt* keywords with variable solution types are read via GmfGetLin fallback.
- Only keywords actually present in the file appear in the output dict.
"""

import ctypes
import ctypes.util
import os
import sys
from typing import Optional, Dict, Any, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# GMF constants (from libmeshb8.h)
# ---------------------------------------------------------------------------

_GMF_READ   = 1
_GMF_WRITE  = 2

_GMF_FLOAT  = 8
_GMF_DOUBLE = 9
_GMF_INT    = 10
_GMF_LONG   = 11

# Solution-field size multipliers
_GMF_SCA     = 1   # scalar
_GMF_VEC     = 2   # vector (dim components)
_GMF_SYMMAT  = 3   # symmetric matrix
_GMF_MAT     = 4   # full matrix

_SOL_SIZE = {_GMF_SCA: 1, _GMF_VEC: None, _GMF_SYMMAT: None, _GMF_MAT: None}


# ---------------------------------------------------------------------------
# Keyword → enum value (from GmfKwdCod enum in libmeshb8.h)
# ---------------------------------------------------------------------------

_KWD_INT: Dict[str, int] = {
    "VersionFormatted":                      1,
    "Dimension":                             3,
    "Vertices":                              4,
    "Edges":                                 5,
    "Triangles":                             6,
    "Quadrilaterals":                        7,
    "Tetrahedra":                            8,
    "Prisms":                                9,
    "Hexahedra":                            10,
    "Corners":                              13,
    "Ridges":                               14,
    "RequiredVertices":                     15,
    "RequiredEdges":                        16,
    "RequiredTriangles":                    17,
    "RequiredQuadrilaterals":               18,
    "TangentAtEdgeVertices":                19,
    "NormalAtVertices":                     20,
    "NormalAtTriangleVertices":             21,
    "NormalAtQuadrilateralVertices":        22,
    "AngleOfCornerBound":                   23,
    "TrianglesP2":                          24,
    "EdgesP2":                              25,
    "SolAtPyramids":                        26,
    "QuadrilateralsQ2":                     27,
    "ISolAtPyramids":                       28,
    "SubDomainFromGeom":                    29,
    "TetrahedraP2":                         30,
    "HexahedraQ2":                          33,
    "ExtraVerticesAtEdges":                 34,
    "ExtraVerticesAtTriangles":             35,
    "ExtraVerticesAtQuadrilaterals":        36,
    "ExtraVerticesAtTetrahedra":            37,
    "ExtraVerticesAtPrisms":                38,
    "ExtraVerticesAtHexahedra":             39,
    "VerticesOnGeometricVertices":          40,
    "VerticesOnGeometricEdges":             41,
    "VerticesOnGeometricTriangles":         42,
    "VerticesOnGeometricQuadrilaterals":    43,
    "EdgesOnGeometricEdges":               44,
    "Polyhedra":                            46,
    "Polygons":                             47,
    "Pyramids":                             49,
    "BoundingBox":                          50,
    "Tangents":                             59,
    "Normals":                              60,
    "TangentAtVertices":                    61,
    "SolAtVertices":                        62,
    "SolAtEdges":                           63,
    "SolAtTriangles":                       64,
    "SolAtQuadrilaterals":                  65,
    "SolAtTetrahedra":                      66,
    "SolAtPrisms":                          67,
    "SolAtHexahedra":                       68,
    "DSolAtVertices":                       69,
    "ISolAtVertices":                       70,
    "ISolAtEdges":                          71,
    "ISolAtTriangles":                      72,
    "ISolAtQuadrilaterals":                 73,
    "ISolAtTetrahedra":                     74,
    "ISolAtPrisms":                         75,
    "ISolAtHexahedra":                      76,
    "Iterations":                           77,
    "Time":                                 78,
    "CoarseHexahedra":                      80,
    "PeriodicVertices":                     82,
    "PeriodicEdges":                        83,
    "PeriodicTriangles":                    84,
    "PeriodicQuadrilaterals":               85,
    "PrismsP2":                             86,
    "PyramidsP2":                           87,
    "QuadrilateralsQ3":                     88,
    "QuadrilateralsQ4":                     89,
    "TrianglesP3":                          90,
    "TrianglesP4":                          91,
    "EdgesP3":                              92,
    "EdgesP4":                              93,
    "TetrahedraP3":                         96,
    "TetrahedraP4":                         97,
    "HexahedraQ3":                          98,
    "HexahedraQ4":                          99,
    "PyramidsP3":                          100,
    "PyramidsP4":                          101,
    "PrismsP3":                            102,
    "PrismsP4":                            103,
    "HOSolAtEdgesP1":                      104,
    "HOSolAtEdgesP2":                      105,
    "HOSolAtEdgesP3":                      106,
    "HOSolAtTrianglesP1":                  107,
    "HOSolAtTrianglesP2":                  108,
    "HOSolAtTrianglesP3":                  109,
    "HOSolAtQuadrilateralsQ1":             110,
    "HOSolAtQuadrilateralsQ2":             111,
    "HOSolAtQuadrilateralsQ3":             112,
    "HOSolAtTetrahedraP1":                 113,
    "HOSolAtTetrahedraP2":                 114,
    "HOSolAtTetrahedraP3":                 115,
    "HOSolAtPyramidsP1":                   116,
    "HOSolAtPyramidsP2":                   117,
    "HOSolAtPyramidsP3":                   118,
    "HOSolAtPrismsP1":                     119,
    "HOSolAtPrismsP2":                     120,
    "HOSolAtPrismsP3":                     121,
    "HOSolAtHexahedraQ1":                  122,
    "HOSolAtHexahedraQ2":                  123,
    "HOSolAtHexahedraQ3":                  124,
    "FloatingPointPrecision":              155,
    "BoundaryLayers":                      198,
    "ReferenceStrings":                    199,
    "Prisms9":                             200,
    "Hexahedra12":                         201,
    "Quadrilaterals6":                     202,
    "Domains":                             209,
    "VerticesGID":                         210,
    "EdgesGID":                            211,
    "TrianglesGID":                        212,
    "QuadrilateralsGID":                   213,
    "TetrahedraGID":                       214,
    "PyramidsGID":                         215,
    "PrismsGID":                           216,
    "HexahedraGID":                        217,
    "SolAtBoundaryPolygons":               218,
    "SolAtPolyhedra":                      219,
    "VerticesOnGeometryNodes":             220,
    "VerticesOnGeometryEdges":             221,
    "EdgesOnGeometryEdges":                222,
    "VerticesOnGeometryFaces":             223,
    "EdgesOnGeometryFaces":                224,
    "TrianglesOnGeometryFaces":            225,
    "MeshOnGeometry":                      227,
}

_KWD_BY_INT: Dict[int, str] = {v: k for k, v in _KWD_INT.items()}


# ---------------------------------------------------------------------------
# Per-keyword field schemas
# ---------------------------------------------------------------------------
# Encoded as a list of (gmf_type, numpy_dtype, ncols, label) tuples.
# "ncols" = number of columns this field occupies in the output array.
# "_DIM_" is a sentinel replaced by the actual mesh dimension at read time.
#
# Keywords absent from this table are silently skipped (unsupported).

_DIM_ = -1  # placeholder for "mesh dimension"

# Entry layout: list of (gmf_type_const, numpy_dtype, ncols, label)
# label is "nodes", "xyz", "ref", or descriptive name.

_SCHEMAS: Dict[str, List[Tuple[int, Any, int, str]]] = {
    # ── Geometry ──────────────────────────────────────────────────────────
    "Vertices":           [(_GMF_DOUBLE, np.float64, _DIM_, "xyz"),
                           (_GMF_INT,    np.int32,   1,     "ref")],

    "Edges":              [(_GMF_INT, np.int32, 2, "nodes"),
                           (_GMF_INT, np.int32, 1, "ref")],

    "Triangles":          [(_GMF_INT, np.int32, 3, "nodes"),
                           (_GMF_INT, np.int32, 1, "ref")],

    "Quadrilaterals":     [(_GMF_INT, np.int32, 4, "nodes"),
                           (_GMF_INT, np.int32, 1, "ref")],

    "Tetrahedra":         [(_GMF_INT, np.int32, 4, "nodes"),
                           (_GMF_INT, np.int32, 1, "ref")],

    "Prisms":             [(_GMF_INT, np.int32, 6, "nodes"),
                           (_GMF_INT, np.int32, 1, "ref")],

    "Hexahedra":          [(_GMF_INT, np.int32, 8, "nodes"),
                           (_GMF_INT, np.int32, 1, "ref")],

    "Pyramids":           [(_GMF_INT, np.int32, 5, "nodes"),
                           (_GMF_INT, np.int32, 1, "ref")],

    # ── Higher-order elements ────────────────────────────────────────────
    "EdgesP2":            [(_GMF_INT, np.int32, 3, "nodes"),
                           (_GMF_INT, np.int32, 1, "ref")],

    "EdgesP3":            [(_GMF_INT, np.int32, 4, "nodes"),
                           (_GMF_INT, np.int32, 1, "ref")],

    "EdgesP4":            [(_GMF_INT, np.int32, 5, "nodes"),
                           (_GMF_INT, np.int32, 1, "ref")],

    "TrianglesP2":        [(_GMF_INT, np.int32, 6, "nodes"),
                           (_GMF_INT, np.int32, 1, "ref")],

    "TrianglesP3":        [(_GMF_INT, np.int32, 10, "nodes"),
                           (_GMF_INT, np.int32, 1,  "ref")],

    "TrianglesP4":        [(_GMF_INT, np.int32, 15, "nodes"),
                           (_GMF_INT, np.int32, 1,  "ref")],

    "QuadrilateralsQ2":   [(_GMF_INT, np.int32, 9, "nodes"),
                           (_GMF_INT, np.int32, 1, "ref")],

    "QuadrilateralsQ3":   [(_GMF_INT, np.int32, 16, "nodes"),
                           (_GMF_INT, np.int32, 1,  "ref")],

    "QuadrilateralsQ4":   [(_GMF_INT, np.int32, 25, "nodes"),
                           (_GMF_INT, np.int32, 1,  "ref")],

    "TetrahedraP2":       [(_GMF_INT, np.int32, 10, "nodes"),
                           (_GMF_INT, np.int32, 1,  "ref")],

    "TetrahedraP3":       [(_GMF_INT, np.int32, 20, "nodes"),
                           (_GMF_INT, np.int32, 1,  "ref")],

    "TetrahedraP4":       [(_GMF_INT, np.int32, 35, "nodes"),
                           (_GMF_INT, np.int32, 1,  "ref")],

    "PyramidsP2":         [(_GMF_INT, np.int32, 14, "nodes"),
                           (_GMF_INT, np.int32, 1,  "ref")],

    "PrismsP2":           [(_GMF_INT, np.int32, 18, "nodes"),
                           (_GMF_INT, np.int32, 1,  "ref")],

    "HexahedraQ2":        [(_GMF_INT, np.int32, 27, "nodes"),
                           (_GMF_INT, np.int32, 1,  "ref")],

    # ── Pure-index keywords (no ref column) ─────────────────────────────
    "Corners":            [(_GMF_INT, np.int32, 1, "nodes")],
    "Ridges":             [(_GMF_INT, np.int32, 1, "nodes")],
    "RequiredVertices":   [(_GMF_INT, np.int32, 1, "nodes")],
    "RequiredEdges":      [(_GMF_INT, np.int32, 1, "nodes")],
    "RequiredTriangles":  [(_GMF_INT, np.int32, 1, "nodes")],
    "RequiredQuadrilaterals": [(_GMF_INT, np.int32, 1, "nodes")],

    # ── Geometric attributes ────────────────────────────────────────────
    "NormalAtVertices":              [(_GMF_INT, np.int32, 2, "data")],
    "TangentAtVertices":             [(_GMF_INT, np.int32, 2, "data")],
    "TangentAtEdgeVertices":         [(_GMF_INT, np.int32, 3, "data")],
    "NormalAtTriangleVertices":      [(_GMF_INT, np.int32, 3, "data")],
    "NormalAtQuadrilateralVertices": [(_GMF_INT, np.int32, 4, "data")],

    "Tangents":           [(_GMF_DOUBLE, np.float64, _DIM_, "xyz"),
                           (_GMF_DOUBLE, np.float64, 1,     "param")],
    "Normals":            [(_GMF_DOUBLE, np.float64, _DIM_, "xyz"),
                           (_GMF_DOUBLE, np.float64, 1,     "param")],

    # ── Periodicity ─────────────────────────────────────────────────────
    "PeriodicVertices":       [(_GMF_INT, np.int32, 2, "data")],
    "PeriodicEdges":          [(_GMF_INT, np.int32, 2, "data")],
    "PeriodicTriangles":      [(_GMF_INT, np.int32, 2, "data")],
    "PeriodicQuadrilaterals": [(_GMF_INT, np.int32, 2, "data")],

    # ── Integer solutions ───────────────────────────────────────────────
    "ISolAtVertices":         [(_GMF_INT, np.int32, 1, "data")],
    "ISolAtEdges":            [(_GMF_INT, np.int32, 2, "data")],
    "ISolAtTriangles":        [(_GMF_INT, np.int32, 3, "data")],
    "ISolAtQuadrilaterals":   [(_GMF_INT, np.int32, 4, "data")],
    "ISolAtTetrahedra":       [(_GMF_INT, np.int32, 4, "data")],
    "ISolAtPrisms":           [(_GMF_INT, np.int32, 6, "data")],
    "ISolAtHexahedra":        [(_GMF_INT, np.int32, 8, "data")],

    # ── Global IDs (int64) ───────────────────────────────────────────────
    "VerticesGID":            [(_GMF_LONG, np.int64, 1, "gid")],
    "EdgesGID":               [(_GMF_LONG, np.int64, 1, "gid")],
    "TrianglesGID":           [(_GMF_LONG, np.int64, 1, "gid")],
    "QuadrilateralsGID":      [(_GMF_LONG, np.int64, 1, "gid")],
    "TetrahedraGID":          [(_GMF_LONG, np.int64, 1, "gid")],
    "PyramidsGID":            [(_GMF_LONG, np.int64, 1, "gid")],
    "PrismsGID":              [(_GMF_LONG, np.int64, 1, "gid")],
    "HexahedraGID":           [(_GMF_LONG, np.int64, 1, "gid")],

    # ── On-geometry tags ─────────────────────────────────────────────────
    "VerticesOnGeometricVertices":          [(_GMF_INT, np.int32, 2, "data")],
    "VerticesOnGeometricEdges":             [(_GMF_INT,    np.int32,   2, "idx"),
                                             (_GMF_DOUBLE, np.float64, 2, "param")],
    "VerticesOnGeometricTriangles":         [(_GMF_INT,    np.int32,   2, "idx"),
                                             (_GMF_DOUBLE, np.float64, 3, "param")],
    "EdgesOnGeometricEdges":                [(_GMF_INT, np.int32, 2, "data")],
    "SubDomainFromGeom":                    [(_GMF_INT, np.int32, 3, "data")],
}

# Keywords whose values are read via SolAtXxx logic (sol header + sol data)
_SOL_KWDS = {
    "SolAtVertices", "SolAtEdges", "SolAtTriangles", "SolAtQuadrilaterals",
    "SolAtTetrahedra", "SolAtPrisms", "SolAtHexahedra",
    "DSolAtVertices",
    "SolAtBoundaryPolygons", "SolAtPolyhedra",
    "HOSolAtEdgesP1", "HOSolAtEdgesP2", "HOSolAtEdgesP3",
    "HOSolAtTrianglesP1", "HOSolAtTrianglesP2", "HOSolAtTrianglesP3",
    "HOSolAtTetrahedraP1", "HOSolAtTetrahedraP2", "HOSolAtTetrahedraP3",
}

# Keywords that are scalar or special (not list-of-elements)
_SCALAR_KWDS = {"Iterations", "Time", "FloatingPointPrecision", "AngleOfCornerBound"}


# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------

_LIB: Optional[ctypes.CDLL] = None


def _load_library() -> ctypes.CDLL:
    """Locate and load libmeshb8 (or compatible) shared library."""
    # 1. Try ctypes.util.find_library
    for stem in ("meshb8", "libmeshb8", "meshb7", "libmeshb7", "meshb", "libmeshb"):
        path = ctypes.util.find_library(stem)
        if path:
            try:
                return ctypes.CDLL(path)
            except OSError:
                pass

    # 2. Explicit search in standard locations
    if sys.platform.startswith("linux"):
        dirs  = ["/usr/local/lib", "/usr/lib", "/usr/lib/x86_64-linux-gnu",
                 "/usr/lib/aarch64-linux-gnu"]
        exts  = [".so"]
    elif sys.platform == "darwin":
        dirs  = ["/usr/local/lib", "/opt/homebrew/lib", "/opt/local/lib"]
        exts  = [".dylib"]
    elif sys.platform.startswith("win"):
        dirs  = ["C:\\Windows\\System32", "C:\\Program Files\\libmeshb\\lib"]
        exts  = [".dll"]
    else:
        dirs, exts = [], []

    for stem in ("libmeshb8", "libmeshb7", "libmeshb"):
        for d in dirs:
            for ext in exts:
                p = os.path.join(d, stem + ext)
                if os.path.exists(p):
                    return ctypes.CDLL(p)

    # 3. Check LIBMESHB_PATH env variable
    env_path = os.environ.get("LIBMESHB_PATH")
    if env_path and os.path.exists(env_path):
        return ctypes.CDLL(env_path)

    raise RuntimeError(
        "libmeshb8 shared library not found.\n"
        "Options:\n"
        "  1. Install from https://github.com/LoicMarechal/libMeshb\n"
        "  2. Set LD_LIBRARY_PATH (Linux) / DYLD_LIBRARY_PATH (macOS) to its location\n"
        "  3. Set the LIBMESHB_PATH environment variable to the full .so/.dylib path"
    )


def _get_lib() -> ctypes.CDLL:
    """Return (and cache) the loaded + configured library handle."""
    global _LIB
    if _LIB is None:
        lib = _load_library()
        # All functions are variadic (or at least have optional extra args),
        # so we leave argtypes=None and rely on explicit ctypes typing.
        lib.GmfOpenMesh.restype  = ctypes.c_int64
        lib.GmfCloseMesh.restype = ctypes.c_int
        lib.GmfStatKwd.restype   = ctypes.c_int64
        lib.GmfGotoKwd.restype   = ctypes.c_int
        lib.GmfGetLin.restype    = ctypes.c_int
        lib.GmfGetBlock.restype  = ctypes.c_int
        _LIB = lib
    return _LIB


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _c_void_p_offset(base_int: int, offset: int) -> ctypes.c_void_p:
    """Return a ctypes void pointer shifted by `offset` bytes from `base_int`."""
    return ctypes.c_void_p(base_int + offset)


def _dtype_size(dtype: np.dtype) -> int:
    return np.dtype(dtype).itemsize


def _gmf_type_size(gmf_type: int) -> int:
    return {_GMF_FLOAT: 4, _GMF_DOUBLE: 8, _GMF_INT: 4, _GMF_LONG: 8}[gmf_type]


def _resolve_schema(schema, dim: int):
    """Replace _DIM_ sentinels with the actual mesh dimension."""
    return [(gt, dt, (dim if nc == _DIM_ else nc), lbl)
            for gt, dt, nc, lbl in schema]


# ---------------------------------------------------------------------------
# GmfGetBlock caller
# ---------------------------------------------------------------------------

def _call_get_block(lib, lib_idx: int, kwd_int: int,
                    n_elem: int, field_specs) -> Dict[str, np.ndarray]:
    """
    Allocate numpy arrays and call GmfGetBlock once for all fields.

    field_specs: resolved schema – list of (gmf_type, np_dtype, ncols, label)

    Returns dict label → ndarray.
    """
    arrays: Dict[str, np.ndarray] = {}
    args: list = []

    for gmf_type, np_dtype, ncols, label in field_specs:
        arr = np.empty((n_elem, ncols) if ncols > 1 else (n_elem,), dtype=np_dtype)
        arr_base = arr.ctypes.data        # Python int (address)
        elem_sz  = _gmf_type_size(gmf_type)
        row_stride = ncols * elem_sz       # bytes per row (= step to next element)

        for col in range(ncols):
            args += [
                ctypes.c_int(gmf_type),
                ctypes.c_size_t(row_stride),
                _c_void_p_offset(arr_base, col * elem_sz),
            ]
        arrays[label] = arr

    # GmfGetBlock(lib_idx, kwd, beg, end, n_callbacks, pre_cbk, pst_cbk, *triplets)
    lib.GmfGetBlock(
        ctypes.c_int64(lib_idx),
        ctypes.c_int(kwd_int),
        ctypes.c_int64(1),
        ctypes.c_int64(n_elem),
        ctypes.c_int(0),          # no callbacks
        None,                      # pre-callback
        None,                      # post-callback
        *args,
    )
    return arrays


# ---------------------------------------------------------------------------
# Sol-keyword reader  (GmfGetLin-based fallback for variable-size fields)
# ---------------------------------------------------------------------------

def _read_sol_kwd(lib, lib_idx: int, kwd_name: str, kwd_int: int,
                  n_elem: int) -> Dict[str, Any]:
    """
    Read a SolAt* keyword.

    GmfStatKwd for sol keywords returns (n_elem, n_types, type_array).
    We first query the solution type descriptor, then read the data.
    """
    # Query: how many solution types, and which?
    MAX_TYPES = 100
    TypTab = (ctypes.c_int * MAX_TYPES)()
    NmbTyp = ctypes.c_int(0)

    lib.GmfStatKwd(
        ctypes.c_int64(lib_idx),
        ctypes.c_int(kwd_int),
        ctypes.byref(NmbTyp),
        TypTab,
    )
    n_types = NmbTyp.value
    sol_types = list(TypTab[:n_types])

    if n_elem == 0 or n_types == 0:
        return {"data": np.empty((0, 0), dtype=np.float64), "sol_types": sol_types}

    # Compute total doubles per element
    # GmfSca=1 → 1, GmfVec=2 → dim, GmfSymMat=3 → dim*(dim+1)/2, GmfMat=4 → dim*dim
    # We read all components as a flat GmfGetLin call; the library handles the expansion.
    # Use GmfGetLin once per element via a pre-allocated buffer.
    MAX_SOL = 512
    buf = (ctypes.c_double * MAX_SOL)()

    # First pass: count actual floats per line by reading one line
    lib.GmfGotoKwd(ctypes.c_int64(lib_idx), ctypes.c_int(kwd_int))
    # We use the raw ctypes approach: pass MAX_SOL double pointers
    # (the library only fills what it needs; the rest is untouched)
    # This is a conservative approach – read element by element.
    sol_rows = []
    for _ in range(n_elem):
        lib.GmfGetLin(
            ctypes.c_int64(lib_idx),
            ctypes.c_int(kwd_int),
            *[ctypes.byref(buf, i * 8) for i in range(MAX_SOL)],
        )
        sol_rows.append(list(buf))  # we trim after first valid row

    # Trim to actual sol size (remaining entries stay 0.0 after last GmfGetLin write)
    # Heuristic: detect trailing zeros after first non-trivial scan
    data = np.array(sol_rows, dtype=np.float64)

    return {
        "data":      data,
        "sol_types": sol_types,
    }


def _read_sol_kwd_block(lib, lib_idx: int, kwd_name: str, kwd_int: int,
                        n_elem: int, dim: int) -> Dict[str, Any]:
    """
    Faster sol reader using GmfGetBlock.
    Computes sol_size from the type descriptor then reads in one call.
    """
    MAX_TYPES = 100
    TypTab = (ctypes.c_int * MAX_TYPES)()
    NmbTyp = ctypes.c_int(0)

    lib.GmfStatKwd(
        ctypes.c_int64(lib_idx),
        ctypes.c_int(kwd_int),
        ctypes.byref(NmbTyp),
        TypTab,
    )
    n_types  = NmbTyp.value
    sol_types = list(TypTab[:n_types])

    if n_elem == 0 or n_types == 0:
        return {"data": np.empty((0, 0), dtype=np.float64), "sol_types": sol_types}

    sol_size_map = {
        _GMF_SCA:    1,
        _GMF_VEC:    dim,
        _GMF_SYMMAT: dim * (dim + 1) // 2,
        _GMF_MAT:    dim * dim,
    }
    sol_size = sum(sol_size_map.get(t, 1) for t in sol_types)
    data = np.empty((n_elem, sol_size), dtype=np.float64)

    base       = data.ctypes.data
    row_stride = sol_size * 8

    args = []
    for col in range(sol_size):
        args += [
            ctypes.c_int(_GMF_DOUBLE),
            ctypes.c_size_t(row_stride),
            _c_void_p_offset(base, col * 8),
        ]

    lib.GmfGetBlock(
        ctypes.c_int64(lib_idx),
        ctypes.c_int(kwd_int),
        ctypes.c_int64(1),
        ctypes.c_int64(n_elem),
        ctypes.c_int(0),
        None,
        None,
        *args,
    )
    return {"data": data, "sol_types": sol_types}


# ---------------------------------------------------------------------------
# Scalar keyword readers
# ---------------------------------------------------------------------------

def _read_scalar_kwd(lib, lib_idx: int, kwd_name: str, kwd_int: int) -> Any:
    """Read single-value keywords like Iterations, Time."""
    if kwd_name in ("Iterations", "FloatingPointPrecision"):
        val = ctypes.c_int(0)
        lib.GmfGotoKwd(ctypes.c_int64(lib_idx), ctypes.c_int(kwd_int))
        lib.GmfGetLin(ctypes.c_int64(lib_idx), ctypes.c_int(kwd_int), ctypes.byref(val))
        return int(val.value)
    elif kwd_name in ("Time", "AngleOfCornerBound"):
        val = ctypes.c_double(0.0)
        lib.GmfGotoKwd(ctypes.c_int64(lib_idx), ctypes.c_int(kwd_int))
        lib.GmfGetLin(ctypes.c_int64(lib_idx), ctypes.c_int(kwd_int), ctypes.byref(val))
        return float(val.value)
    return None


def _read_bounding_box(lib, lib_idx: int, kwd_int: int, dim: int
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """Read BoundingBox → (lo, hi) each shape (dim,) float64."""
    vals = (ctypes.c_double * (2 * dim))()
    lib.GmfGotoKwd(ctypes.c_int64(lib_idx), ctypes.c_int(kwd_int))
    lib.GmfGetLin(
        ctypes.c_int64(lib_idx),
        ctypes.c_int(kwd_int),
        *[ctypes.byref(vals, i * 8) for i in range(2 * dim)],
    )
    arr = np.frombuffer(vals, dtype=np.float64).copy()
    return arr[:dim], arr[dim:]


# ---------------------------------------------------------------------------
# Main reader
# ---------------------------------------------------------------------------

def read_mesh(filepath: str,
              keywords: Optional[List[str]] = None,
              verbose: bool = False) -> Dict[str, Any]:
    """
    Read a GMF mesh file into a dictionary.

    Parameters
    ----------
    filepath : str
        Path to a .mesh (ASCII) or .meshb (binary) file.
    keywords : list of str, optional
        If given, only read these keywords (e.g. ["Vertices", "Triangles"]).
        By default every keyword present in the file is read.
    verbose : bool
        Print progress information.

    Returns
    -------
    dict
        Keys are GMF keyword names (without the "Gmf" prefix).
        Special keys:
          "Dimension" → int
          "Version"   → int
          All others  → see module docstring for layout.

    Raises
    ------
    FileNotFoundError
        If `filepath` does not exist.
    RuntimeError
        If the library cannot be found or the file cannot be opened.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Mesh file not found: {filepath}")

    lib = _get_lib()

    # ── Open the file ──────────────────────────────────────────────────────
    ver = ctypes.c_int(0)
    dim = ctypes.c_int(0)
    lib_idx = lib.GmfOpenMesh(
        filepath.encode(),
        ctypes.c_int(_GMF_READ),
        ctypes.byref(ver),
        ctypes.byref(dim),
    )
    if lib_idx == 0:
        raise RuntimeError(f"GmfOpenMesh failed for '{filepath}'. "
                           "Check that the file is a valid .mesh/.meshb.")

    version   = int(ver.value)
    dimension = int(dim.value)

    mesh: Dict[str, Any] = {
        "Version":   version,
        "Dimension": dimension,
    }

    if verbose:
        print(f"[pymeshb] Opened '{filepath}'  version={version}  dim={dimension}")

    try:
        # ── Discover which keywords are present ────────────────────────────
        all_kwds = keywords if keywords is not None else list(_KWD_INT.keys())

        for kwd_name in all_kwds:
            kwd_int = _KWD_INT.get(kwd_name)
            if kwd_int is None:
                continue  # unknown keyword

            n_elem = int(lib.GmfStatKwd(
                ctypes.c_int64(lib_idx),
                ctypes.c_int(kwd_int),
            ))

            if n_elem == 0:
                continue  # not present in file

            if verbose:
                print(f"[pymeshb]   {kwd_name}: {n_elem} element(s)")

            # ── Scalar keywords ────────────────────────────────────────────
            if kwd_name in _SCALAR_KWDS:
                mesh[kwd_name] = _read_scalar_kwd(lib, lib_idx, kwd_name, kwd_int)
                continue

            if kwd_name == "BoundingBox":
                lo, hi = _read_bounding_box(lib, lib_idx, kwd_int, dimension)
                mesh["BoundingBox"] = {"lo": lo, "hi": hi}
                continue

            # ── Solution keywords (variable-width) ─────────────────────────
            if kwd_name in _SOL_KWDS:
                mesh[kwd_name] = _read_sol_kwd_block(
                    lib, lib_idx, kwd_name, kwd_int, n_elem, dimension
                )
                continue

            # ── Regular list-of-elements keywords ─────────────────────────
            raw_schema = _SCHEMAS.get(kwd_name)
            if raw_schema is None:
                if verbose:
                    print(f"[pymeshb]   → no schema for '{kwd_name}', skipping")
                continue

            schema = _resolve_schema(raw_schema, dimension)

            # Goto is not strictly needed for GmfGetBlock, but ensures
            # file pointer is positioned (some file versions need it).
            lib.GmfGotoKwd(ctypes.c_int64(lib_idx), ctypes.c_int(kwd_int))

            fields = _call_get_block(lib, lib_idx, kwd_int, n_elem, schema)

            # Post-process: flatten single-column arrays to 1-D
            result: Dict[str, Any] = {}
            for label, arr in fields.items():
                if arr.ndim == 2 and arr.shape[1] == 1:
                    result[label] = arr[:, 0]
                else:
                    result[label] = arr

            # If only one field group, unwrap to just the array for simple keywords
            # (e.g. Corners → just int32 array of vertex indices)
            if len(result) == 1:
                mesh[kwd_name] = next(iter(result.values()))
            else:
                mesh[kwd_name] = result

    finally:
        lib.GmfCloseMesh(ctypes.c_int64(lib_idx))

    return mesh


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def list_keywords(filepath: str) -> List[str]:
    """
    Return the list of keyword names that are present in the file
    (without reading their data).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Mesh file not found: {filepath}")

    lib = _get_lib()
    ver = ctypes.c_int(0)
    dim = ctypes.c_int(0)
    lib_idx = lib.GmfOpenMesh(
        filepath.encode(),
        ctypes.c_int(_GMF_READ),
        ctypes.byref(ver),
        ctypes.byref(dim),
    )
    if lib_idx == 0:
        raise RuntimeError(f"GmfOpenMesh failed for '{filepath}'.")

    found = []
    try:
        for kwd_name, kwd_int in _KWD_INT.items():
            n = int(lib.GmfStatKwd(
                ctypes.c_int64(lib_idx),
                ctypes.c_int(kwd_int),
            ))
            if n > 0:
                found.append(kwd_name)
    finally:
        lib.GmfCloseMesh(ctypes.c_int64(lib_idx))

    return found


def mesh_summary(mesh: Dict[str, Any]) -> str:
    """Return a human-readable summary string of a mesh dict."""
    lines = [
        f"Dimension : {mesh.get('Dimension')}",
        f"Version   : {mesh.get('Version')}",
    ]
    skip = {"Dimension", "Version"}
    for k, v in mesh.items():
        if k in skip:
            continue
        if isinstance(v, np.ndarray):
            lines.append(f"{k:30s}: {v.shape}  dtype={v.dtype}")
        elif isinstance(v, dict):
            shapes = {lbl: arr.shape for lbl, arr in v.items() if isinstance(arr, np.ndarray)}
            lines.append(f"{k:30s}: {shapes}")
        else:
            lines.append(f"{k:30s}: {v}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Solution field splitter
# ---------------------------------------------------------------------------

#: Human-readable names for GMF solution types.
SOL_TYPE_NAMES = {
    _GMF_SCA:    "scalar",
    _GMF_VEC:    "vector",
    _GMF_SYMMAT: "symmetric_matrix",
    _GMF_MAT:    "matrix",
}


def sol_field_sizes(sol_types: List[int], dim: int) -> List[int]:
    """
    Return the number of components for each field given sol_types and dim.

    Parameters
    ----------
    sol_types : list of int
        As stored in mesh[kwd]["sol_types"]  (GmfSca=1, GmfVec=2, …)
    dim : int
        Mesh dimension (2 or 3).

    Returns
    -------
    list of int  – one entry per field.

    Examples
    --------
    >>> sol_field_sizes([1, 2, 3], dim=3)
    [1, 3, 6]   # scalar, vector(3), sym-matrix(6)
    """
    size_map = {
        _GMF_SCA:    1,
        _GMF_VEC:    dim,
        _GMF_SYMMAT: dim * (dim + 1) // 2,
        _GMF_MAT:    dim * dim,
    }
    sizes = []
    for t in sol_types:
        s = size_map.get(t)
        if s is None:
            raise ValueError(f"Unknown sol_type value {t}. "
                             f"Expected one of {list(size_map)}.")
        sizes.append(s)
    return sizes


def split_sol(
    sol: Dict[str, Any],
    dim: int,
    names: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """
    Split a packed SolAt* array into individual named fields.

    Parameters
    ----------
    sol : dict
        The dict returned by read_mesh for a SolAt* keyword, e.g.
        mesh["SolAtVertices"].  Must have keys "data" and "sol_types".
    dim : int
        Mesh dimension – use mesh["Dimension"].
    names : list of str, optional
        One name per field.  If omitted (or shorter than the number of
        fields) the remaining fields are named "field_0", "field_1", …

    Returns
    -------
    dict  – { field_name: np.ndarray of shape (N,) or (N, k) }
        Scalar fields (k=1) are returned as shape (N,) for convenience.

    Examples
    --------
    Read a .solb with Density (scalar) + Velocity (vector) in 3D:

        mesh = read_mesh("flow.solb")
        fields = split_sol(mesh["SolAtVertices"], dim=mesh["Dimension"],
                           names=["Density", "Velocity"])
        fields["Density"]   # (N,)   float64
        fields["Velocity"]  # (N, 3) float64

    If you don't know the field names yet:

        fields = split_sol(mesh["SolAtVertices"], dim=3)
        # keys: "field_0", "field_1", …
    """
    data      = sol["data"]          # (N, total_cols)
    sol_types = sol["sol_types"]

    if data.ndim == 1:
        data = data[:, np.newaxis]

    sizes  = sol_field_sizes(sol_types, dim)
    n_flds = len(sizes)

    # Build / pad the names list
    if names is None:
        names = []
    names = list(names)
    for i in range(len(names), n_flds):
        names.append(f"field_{i}")

    if len(names) > n_flds:
        raise ValueError(
            f"{len(names)} names supplied but only {n_flds} field(s) in sol_types."
        )

    total_expected = sum(sizes)
    if data.shape[1] != total_expected:
        raise ValueError(
            f"data has {data.shape[1]} column(s) but sol_types {sol_types} "
            f"with dim={dim} require {total_expected} column(s)."
        )

    result: Dict[str, np.ndarray] = {}
    col = 0
    for name, size, sol_type in zip(names, sizes, sol_types):
        chunk = data[:, col : col + size]
        # Squeeze scalar fields to 1-D
        result[name] = chunk[:, 0] if size == 1 else chunk
        col += size

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pymeshb.py <file.mesh[b]> [keyword1 keyword2 ...]")
        sys.exit(1)

    fpath = sys.argv[1]

    kwds = sys.argv[2:] if len(sys.argv) > 2 else None

    print(f"Scanning keywords in '{fpath}' ...")
    present = list_keywords(fpath)
    print("Keywords present:", ", ".join(present))
    print()

    print("Reading mesh ...")
    m = read_mesh(fpath, keywords=kwds, verbose=True)
    print()
    print(mesh_summary(m))
