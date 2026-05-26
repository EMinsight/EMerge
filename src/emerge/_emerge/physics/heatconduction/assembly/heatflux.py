import numpy as np
from ....elements.leg2 import Legrange2
from scipy.sparse import csr_matrix
from ....mth.optimized import local_mapping, matinv, compute_distances, gaus_quad_tet
from numba import c16, types, f8, i8, njit, prange
from .grad import tet_coefficients


############################################################
#               TRIANGLE AND VOLUME GQ POINTS              #
############################################################
# fmt: off
TRI_DPTS = np.array([
    [-0.28125,     0.26041667,  0.26041667,  0.26041667],
    [ 1.0/3.0,     0.6,         0.2,         0.2        ],
    [ 1.0/3.0,     0.2,         0.6,         0.2        ],
    [ 1.0/3.0,     0.2,         0.2,         0.6        ],
], dtype=np.float64)

DPTS = np.array([
    [-0.078933,    0.04573333,  0.04573333,  0.04573333,  0.04573333,
      0.14933333,  0.14933333,  0.14933333,  0.14933333,  0.14933333,  0.14933333],
    [ 0.25,        0.78571429,  0.07142857,  0.07142857,  0.07142857,
      0.39940358,  0.39940358,  0.39940358,  0.10059642,  0.10059642,  0.10059642],
    [ 0.25,        0.07142857,  0.07142857,  0.07142857,  0.78571429,
      0.10059642,  0.10059642,  0.39940358,  0.39940358,  0.39940358,  0.10059642],
    [ 0.25,        0.07142857,  0.07142857,  0.78571429,  0.07142857,
      0.39940358,  0.10059642,  0.10059642,  0.39940358,  0.10059642,  0.39940358],
    [ 0.25,        0.07142857,  0.78571429,  0.07142857,  0.07142857,
      0.10059642,  0.39940358,  0.10059642,  0.10059642,  0.39940358,  0.39940358],
], dtype=np.float64)
# fmt: on


############################################################
#                   SURFACE HEAT FLUX BC                  #
############################################################


@njit(i8[:, :](i8[:], i8[:, :]), cache=True, nogil=True)
def _local_tri_edge_map(vert_ids, edge_verts):
    """Map global edge vertex pairs to local triangle vertex indices 0,1,2."""
    nedges = edge_verts.shape[1]
    out = np.zeros((2, nedges), dtype=np.int64)
    for ie in range(nedges):
        for row in range(2):
            gid = edge_verts[row, ie]
            for k in range(3):
                if vert_ids[k] == gid:
                    out[row, ie] = k
                    break
    return out


@njit(
    f8[:](f8[:, :], i8[:, :], i8[:, :], i8[:, :], i8[:, :], i8[:, :], f8, i8),
    cache=True,
    nogil=True,
    parallel=True,
)
def _surface_flux_builder(
    nodes, tris, edges, tri_to_field, tri_ids_2d, local_edge_maps, q_flux, n_field
):
    """Parallel assembly of surface heat flux load vector.

    Args:
        nodes: (3, n_nodes)
        tris: (3, n_tris_total)
        edges: (2, n_edges)
        tri_to_field: (6, n_tris_total)
        tri_ids_2d: (1, n_selected) selected triangle indices (2D for numba)
        local_edge_maps: (2*n_selected, 3) precomputed local edge maps, stacked
        q_flux: constant heat flux [W/m²]
        n_field: total number of DOFs

    Returns:
        f_global: (n_field,) load vector
    """
    n_sel = tri_ids_2d.shape[1]

    # Per-thread accumulation buffers
    n_threads = 6
    f_threads = np.zeros((n_threads, n_field), dtype=np.float64)

    weights = TRI_DPTS[0, :]
    nq = weights.shape[0]

    for idx in prange(n_sel):
        tid = idx % n_threads
        itri = tri_ids_2d[0, idx]

        # Triangle vertex coordinates
        v0 = tris[0, itri]
        v1 = tris[1, itri]
        v2 = tris[2, itri]

        # Compute area via cross product
        e01x = nodes[0, v1] - nodes[0, v0]
        e01y = nodes[1, v1] - nodes[1, v0]
        e01z = nodes[2, v1] - nodes[2, v0]
        e02x = nodes[0, v2] - nodes[0, v0]
        e02y = nodes[1, v2] - nodes[1, v0]
        e02z = nodes[2, v2] - nodes[2, v0]
        cx = e01y * e02z - e01z * e02y
        cy = e01z * e02x - e01x * e02z
        cz = e01x * e02y - e01y * e02x
        area = 0.5 * np.sqrt(cx * cx + cy * cy + cz * cz)

        # Local edge map for this triangle
        lem = local_edge_maps[2 * idx : 2 * idx + 2, :]

        # Field DOF indices
        field_ids = tri_to_field[:, itri]

        # Quadrature
        f_local = np.zeros(6, dtype=np.float64)

        for iq in range(nq):
            L1 = TRI_DPTS[1, iq]
            L2 = TRI_DPTS[2, iq]
            L3 = TRI_DPTS[3, iq]
            w = weights[iq]

            Ls = np.empty(3, dtype=np.float64)
            Ls[0] = L1
            Ls[1] = L2
            Ls[2] = L3

            # Vertex basis: N_i = L_i(2*L_i - 1)
            for iv in range(3):
                f_local[iv] += w * Ls[iv] * (2.0 * Ls[iv] - 1.0)

            # Edge midpoint basis: N_ij = 4*L_i*L_j
            for ie in range(3):
                li = Ls[lem[0, ie]]
                lj = Ls[lem[1, ie]]
                f_local[3 + ie] += w * 4.0 * li * lj

        # Scale: weights sum to 1/2 (ref tri area), so multiply by 2*area*q
        scale = q_flux * 2.0 * area

        for i in range(6):
            f_threads[tid, field_ids[i]] += f_local[i] * scale

    # Reduce threads
    f_global = np.zeros(n_field, dtype=np.float64)
    for t in range(n_threads):
        for i in range(n_field):
            f_global[i] += f_threads[t, i]

    return f_global


def assemble_surface_flux(field, mesh, face_tags, q_flux):
    """Assemble surface Neumann heat flux into global load vector.

    Args:
        field: Legrange2 field
        mesh: Mesh3D
        face_tags: list/array of GMSH face tags for this BC
        q_flux: float, constant heat flux [W/m²], positive = into domain

    Returns:
        f_global: (n_field,) load vector contribution
    """
    tri_ids = mesh.get_triangles(face_tags)
    n_sel = len(tri_ids)
    nnodes = field.nnodes

    # Precompute all local edge maps
    local_edge_maps = np.empty((2 * n_sel, 3), dtype=np.int64)
    for i in range(n_sel):
        itri = tri_ids[i]
        vert_ids = mesh.tris[:, itri]
        edge_field_ids = field.tri_to_field[3:6, itri]
        edge_mesh_ids = edge_field_ids - nnodes
        edge_verts = mesh.edges[:, edge_mesh_ids]
        local_edge_maps[2 * i : 2 * i + 2, :] = _local_tri_edge_map(
            vert_ids, edge_verts
        )

    tri_ids_2d = tri_ids.reshape(1, -1).astype(np.int64)

    return _surface_flux_builder(
        mesh.nodes,
        mesh.tris,
        mesh.edges,
        field.tri_to_field,
        tri_ids_2d,
        local_edge_maps,
        float(q_flux),
        field.n_field,
    )


# ============================================================
#  Volumetric heat source
# ============================================================


@njit(
    f8[:](f8[:, :], i8[:, :], i8[:, :], i8[:, :], f8[:, :], i8[:, :], i8),
    cache=True,
    nogil=True,
    parallel=True,
)
def _volume_source_builder(
    nodes, tets, edges, tet_to_field, q_at_quad, local_edge_maps, n_field
):
    """Parallel assembly of volumetric heat source load vector.

    Args:
        nodes: (3, n_nodes)
        tets: (4, n_tets_total)
        edges: (2, n_edges)
        tet_to_field: (10, n_tets_total)
        q_at_quad: (n_selected, n_quad) heat source values at quadrature points
        local_edge_maps: (2*n_selected, 6) precomputed local edge maps, stacked
        n_field: total number of DOFs

    Returns:
        f_global: (n_field,) load vector
    """
    n_sel = q_at_quad.shape[0]
    nq = DPTS.shape[1]
    weights = DPTS[0, :]

    n_threads = 6
    f_threads = np.zeros((n_threads, n_field), dtype=np.float64)

    for idx in prange(n_sel):
        tid = idx % n_threads

        # Tet vertex coordinates
        itet_field = tet_to_field[:, idx]  # not needed yet, see below

        # We need the actual tet index — stored in q_at_quad ordering
        # The caller maps idx -> itet, so tet_to_field columns must match
        # We receive tet_to_field already sliced to selected tets

        v0 = tet_to_field[0, idx]
        v1 = tet_to_field[1, idx]
        v2 = tet_to_field[2, idx]
        v3 = tet_to_field[3, idx]

        # Tet vertices from node coordinates (vertex field IDs = node IDs for Lagrange)
        txs = np.empty(4, dtype=np.float64)
        tys = np.empty(4, dtype=np.float64)
        tzs = np.empty(4, dtype=np.float64)
        txs[0] = nodes[0, v0]
        txs[1] = nodes[0, v1]
        txs[2] = nodes[0, v2]
        txs[3] = nodes[0, v3]
        tys[0] = nodes[1, v0]
        tys[1] = nodes[1, v1]
        tys[2] = nodes[1, v2]
        tys[3] = nodes[1, v3]
        tzs[0] = nodes[2, v0]
        tzs[1] = nodes[2, v1]
        tzs[2] = nodes[2, v2]
        tzs[3] = nodes[2, v3]

        # Barycentric coefficients
        aas, bbs, ccs, dds, V = tet_coefficients(txs, tys, tzs)
        inv6V = 1.0 / (6.0 * V)

        # Normalized coefficients
        a = aas * inv6V
        b = bbs * inv6V
        c = ccs * inv6V
        d = dds * inv6V

        # Physical quadrature coordinates
        L1_pts = DPTS[1, :]
        L2_pts = DPTS[2, :]
        L3_pts = DPTS[3, :]
        L4_pts = DPTS[4, :]

        # Local edge map
        lem = local_edge_maps[2 * idx : 2 * idx + 2, :]

        # Field DOFs for this tet
        field_ids = tet_to_field[:, idx]

        f_local = np.zeros(10, dtype=np.float64)

        for iq in range(nq):
            # Physical coordinate
            x = (
                txs[0] * L1_pts[iq]
                + txs[1] * L2_pts[iq]
                + txs[2] * L3_pts[iq]
                + txs[3] * L4_pts[iq]
            )
            y = (
                tys[0] * L1_pts[iq]
                + tys[1] * L2_pts[iq]
                + tys[2] * L3_pts[iq]
                + tys[3] * L4_pts[iq]
            )
            z = (
                tzs[0] * L1_pts[iq]
                + tzs[1] * L2_pts[iq]
                + tzs[2] * L3_pts[iq]
                + tzs[3] * L4_pts[iq]
            )

            # Barycentric coordinates
            Ls = np.empty(4, dtype=np.float64)
            for iv in range(4):
                Ls[iv] = a[iv] + b[iv] * x + c[iv] * y + d[iv] * z

            wq = weights[iq] * q_at_quad[idx, iq]

            # Vertex basis: N_i = L_i(2*L_i - 1)
            for iv in range(4):
                f_local[iv] += wq * Ls[iv] * (2.0 * Ls[iv] - 1.0)

            # Edge midpoint basis: N_ij = 4*L_i*L_j
            for ie in range(6):
                li = Ls[lem[0, ie]]
                lj = Ls[lem[1, ie]]
                f_local[4 + ie] += wq * 4.0 * li * lj

        # Scale by 6V (weights sum to 1/6)
        scale = V
        for i in range(10):
            f_threads[tid, field_ids[i]] += f_local[i] * scale

    # Reduce threads
    f_global = np.zeros(n_field, dtype=np.float64)
    for t in range(n_threads):
        for i in range(n_field):
            f_global[i] += f_threads[t, i]

    return f_global


def assemble_volume_source(field, tet_ids, q_func):
    """Assemble volumetric heat source into global load vector.

    Args:
        field: Legrange2 field
        tet_ids: array of tet indices where source is active
        q_func: callable(xs, ys, zs) -> array of q_V values [W/m³]

    Returns:
        f_global: (n_field,) load vector contribution
    """
    mesh = field.mesh
    tet_ids = np.asarray(tet_ids, dtype=np.int64)
    n_sel = len(tet_ids)
    nq = DPTS.shape[1]
    nnodes = field.nnodes

    L1 = DPTS[1, :]
    L2 = DPTS[2, :]
    L3 = DPTS[3, :]
    L4 = DPTS[4, :]

    verts = mesh.tets[:, tet_ids]  # (4, n_sel)

    x0, x1, x2, x3 = (
        mesh.nodes[0, verts[0]],
        mesh.nodes[0, verts[1]],
        mesh.nodes[0, verts[2]],
        mesh.nodes[0, verts[3]],
    )
    y0, y1, y2, y3 = (
        mesh.nodes[1, verts[0]],
        mesh.nodes[1, verts[1]],
        mesh.nodes[1, verts[2]],
        mesh.nodes[1, verts[3]],
    )
    z0, z1, z2, z3 = (
        mesh.nodes[2, verts[0]],
        mesh.nodes[2, verts[1]],
        mesh.nodes[2, verts[2]],
        mesh.nodes[2, verts[3]],
    )

    all_xs = x0[:, None] * L1 + x1[:, None] * L2 + x2[:, None] * L3 + x3[:, None] * L4
    all_ys = y0[:, None] * L1 + y1[:, None] * L2 + y2[:, None] * L3 + y3[:, None] * L4
    all_zs = z0[:, None] * L1 + z1[:, None] * L2 + z2[:, None] * L3 + z3[:, None] * L4

    q_at_quad = q_func(all_xs.ravel(), all_ys.ravel(), all_zs.ravel()).reshape(
        n_sel, nq
    )

    # Precompute local edge maps using existing utility
    local_edge_maps = np.empty((2 * n_sel, 6), dtype=np.int64)
    for i in range(n_sel):
        lem = np.array([[0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3]])
        local_edge_maps[2 * i] = lem[0]
        local_edge_maps[2 * i + 1] = lem[1]

    tet_to_field_sel = field.tet_to_field[:, tet_ids].copy()

    return _volume_source_builder(
        mesh.nodes,
        mesh.tets,
        mesh.edges,
        tet_to_field_sel,
        q_at_quad,
        local_edge_maps,
        field.n_field,
    )
