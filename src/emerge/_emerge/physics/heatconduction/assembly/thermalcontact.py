import numpy as np
from numba import njit, f8, i8, types, prange
from .heatflux import TRI_DPTS, _local_tri_edge_map


# @njit(
#     types.Tuple((f8[:], i8[:], i8[:]))(
#         f8[:, :], i8[:, :], i8[:, :], i8[:, :], i8[:, :], i8[:], i8[:, :], f8, i8
#     ),
#     cache=True,
#     nogil=True,
#     parallel=False,
# )
def _thermal_contact_builder(
    nodes,
    tris,
    edges,
    tri_to_field_a,
    tri_to_field_b,
    tri_ids,
    local_edge_maps,
    h_c,
    n_field,
):
    """Parallel assembly of thermal contact coupling in COO format.

    For each face, assembles the surface mass matrix M_ij = h_c * integral N_i N_j dS
    and produces four blocks:
        +M at (A, A)
        +M at (B, B)
        -M at (A, B)
        -M at (B, A)

    Returns:
        values, rows, cols: COO triplets (length n_selected * 144)
    """
    n_triangles = tri_ids.shape[0]
    nnz = n_triangles * 144  # 4 blocks of 6x6 per face

    values = np.empty(nnz, dtype=np.float64)
    rows = np.empty(nnz, dtype=np.int64)
    cols = np.empty(nnz, dtype=np.int64)

    weights = TRI_DPTS[0, :]
    nq = weights.shape[0]

    for idx in prange(n_triangles):
        p = idx * 144
        itri = tri_ids[idx]

        # Triangle area
        v0 = tris[0, itri]
        v1 = tris[1, itri]
        v2 = tris[2, itri]

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

        # Local edge map
        lem0 = local_edge_maps[2 * idx, :]
        lem1 = local_edge_maps[2 * idx + 1, :]

        # Surface mass matrix M_ij = integral N_i N_j dS
        M = np.zeros((6, 6), dtype=np.float64)

        for iq in range(nq):
            L1 = TRI_DPTS[1, iq]
            L2 = TRI_DPTS[2, iq]
            L3 = TRI_DPTS[3, iq]
            w = weights[iq]

            Ls = np.empty(3, dtype=np.float64)
            Ls[0] = L1
            Ls[1] = L2
            Ls[2] = L3

            N = np.empty(6, dtype=np.float64)

            for iv in range(3):
                N[iv] = Ls[iv] * (2.0 * Ls[iv] - 1.0)

            for ie in range(3):
                li = Ls[lem0[ie]]
                lj = Ls[lem1[ie]]
                N[3 + ie] = 4.0 * li * lj

            for i in range(6):
                for j in range(6):
                    M[i, j] += w * N[i] * N[j]

        # Scale: weights sum to 1/2, so multiply by 2*area*h_c
        scale = h_c * 2.0 * area

        # DOF indices for both sides
        fids_a = tri_to_field_a[:, idx]
        fids_b = tri_to_field_b[:, idx]

        # Write 4 blocks of 6x6
        for i in range(6):
            for j in range(6):
                mij = M[i, j] * scale

                # +M at (A, A)
                k = p + 6 * i + j
                rows[k] = fids_a[i]
                cols[k] = fids_a[j]
                values[k] = mij

                # +M at (B, B)
                k = p + 36 + 6 * i + j
                rows[k] = fids_b[i]
                cols[k] = fids_b[j]
                values[k] = mij

                # -M at (A, B)
                k = p + 72 + 6 * i + j
                rows[k] = fids_a[i]
                cols[k] = fids_b[j]
                values[k] = -mij

                # -M at (B, A)
                k = p + 108 + 6 * i + j
                rows[k] = fids_b[i]
                cols[k] = fids_a[j]
                values[k] = -mij

    return values, rows, cols


def assemble_thermal_contact(field, face_tags, h_c):
    """Assemble thermal contact coupling between two volume regions.

    Args:
        field: Legrange2 field (with _dof_mapping populated)
        face_tags: GMSH face tags of the contact interface
        h_c: thermal contact conductance [W/(m²·K)]

    Returns:
        K_values: COO values for stiffness matrix addition
        K_rows: COO row indices
        K_cols: COO column indices
    """
    mesh = field.mesh
    tri_ids = mesh.get_triangles(face_tags)
    n_triangles = len(tri_ids)
    nnodes = field.nnodes

    # Build the DOF mapping arrays from dict
    dof_map = field._dof_mapping

    # Build tri_to_field for A side (original) and B side (remapped)
    tri_to_field_a = np.empty((6, n_triangles), dtype=np.int64)
    tri_to_field_b = np.empty((6, n_triangles), dtype=np.int64)

    local_edge_maps = np.empty((2 * n_triangles, 3), dtype=np.int64)

    for i in range(n_triangles):
        itri = tri_ids[i]

        # A-side DOFs are the original tri_to_field entries
        fids_a = field.tri_to_field[:, itri].copy()
        tri_to_field_a[:, i] = fids_a

        # B-side DOFs: look up each A DOF in the mapping
        fids_b = np.empty(6, dtype=np.int64)
        for j in range(6):
            fids_b[j] = dof_map[int(fids_a[j])]
        tri_to_field_b[:, i] = fids_b

        # Local edge map (same geometry for both sides)
        vert_ids = mesh.tris[:, itri]
        edge_field_ids = fids_a[3:6]
        edge_mesh_ids = edge_field_ids - nnodes
        edge_verts = mesh.edges[:, edge_mesh_ids]
        local_edge_maps[2 * i : 2 * i + 2, :] = _local_tri_edge_map(
            vert_ids, edge_verts
        )

    return _thermal_contact_builder(
        mesh.nodes,
        mesh.tris,
        mesh.edges,
        tri_to_field_a,
        tri_to_field_b,
        tri_ids,
        local_edge_maps,
        float(h_c),
        field.n_field,
    )
