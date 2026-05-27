import numpy as np
from numba import njit, f8, i8, types, prange
from .heatflux import TRI_DPTS, _local_tri_edge_map

# ============================================================
#  Thin Conductive Sheet (ConductiveSheet BC)
#
#  Assembles the 2D surface stiffness matrix:
#    K_ij += integral_Gamma (kappa * t) * grad_s N_i . grad_s N_j dS
#
#  where grad_s is the surface gradient (gradient projected onto
#  the triangle plane) and t is the sheet thickness.
#
#  This models lateral heat spreading in a thin metallic layer
#  (e.g. copper foil on PCB) without needing volume elements.
# ============================================================


@njit(
    types.Tuple((f8[:], i8[:], i8[:]))(
        f8[:, :], i8[:, :], i8[:, :], i8[:, :], i8[:, :], i8[:, :], f8, i8
    ),
    cache=True,
    nogil=True,
    parallel=True,
)
def _conductive_sheet_builder(
    nodes, tris, edges, tri_to_field, tri_ids_2d, local_edge_maps, kappa_t, n_field
):
    """Parallel assembly of thin conductive sheet stiffness in COO format.

    The surface gradient of P2 basis functions on a flat triangle is:
      grad_s N_i = (dN_i/dxi) * e1_hat/|e1| + (dN_i/deta) * e2_hat/|e2|

    But it's easier to work in the physical triangle's tangent plane directly.
    For a triangle with vertices (x1,y1,z1), (x2,y2,z2), (x3,y3,z3):
      The barycentric coords L1,L2,L3 satisfy L1+L2+L3=1
      grad_s L_i is the projection of the 3D gradient onto the triangle plane

    We compute this via the edge vectors and the contravariant basis.

    Args:
        nodes: (3, n_nodes)
        tris: (3, n_tris_total)
        edges: (2, n_edges)
        tri_to_field: (6, n_tris_total)
        tri_ids_2d: (1, n_selected)
        local_edge_maps: (2*n_selected, 3)
        kappa_t: kappa * thickness [W/K] (conductivity times sheet thickness)
        n_field: total DOFs

    Returns:
        values, rows, cols: COO triplets (length n_selected * 36)
    """
    n_sel = tri_ids_2d.shape[1]
    nnz = n_sel * 36  # 6x6 per face

    values = np.empty(nnz, dtype=np.float64)
    rows = np.empty(nnz, dtype=np.int64)
    cols = np.empty(nnz, dtype=np.int64)

    weights = TRI_DPTS[0, :]
    nq = weights.shape[0]

    for idx in prange(n_sel):
        p = idx * 36
        itri = tri_ids_2d[0, idx]

        # Triangle vertex coordinates
        v0 = tris[0, itri]
        v1 = tris[1, itri]
        v2 = tris[2, itri]

        x0 = nodes[0, v0]
        y0 = nodes[1, v0]
        z0 = nodes[2, v0]
        x1 = nodes[0, v1]
        y1 = nodes[1, v1]
        z1 = nodes[2, v1]
        x2 = nodes[0, v2]
        y2 = nodes[1, v2]
        z2 = nodes[2, v2]

        # Edge vectors
        e1x = x1 - x0
        e1y = y1 - y0
        e1z = z1 - z0
        e2x = x2 - x0
        e2y = y2 - y0
        e2z = z2 - z0

        # Normal = e1 x e2
        nx = e1y * e2z - e1z * e2y
        ny = e1z * e2x - e1x * e2z
        nz = e1x * e2y - e1y * e2x
        area2 = np.sqrt(nx * nx + ny * ny + nz * nz)  # 2 * area
        area = 0.5 * area2

        # Metric tensor components: g_ij = e_i . e_j
        g11 = e1x * e1x + e1y * e1y + e1z * e1z
        g12 = e1x * e2x + e1y * e2y + e1z * e2z
        g22 = e2x * e2x + e2y * e2y + e2z * e2z

        # Inverse metric * (2*area)^2 for contravariant basis
        # g^11 = g22/det, g^12 = -g12/det, g^22 = g11/det
        # where det = g11*g22 - g12*g12 = (2*area)^2
        det_g = g11 * g22 - g12 * g12
        inv_det = 1.0 / det_g

        # Surface gradients of barycentric coordinates L1, L2, L3
        # In parametric coords (xi, eta): L1 = 1 - xi - eta, L2 = xi, L3 = eta
        # dL1/dxi = -1, dL1/deta = -1
        # dL2/dxi =  1, dL2/deta =  0
        # dL3/dxi =  0, dL3/deta =  1
        #
        # Surface gradient: grad_s L = (dL/dxi) * g^1 + (dL/deta) * g^2
        # where g^1, g^2 are contravariant basis vectors:
        #   g^1 = (g22*e1 - g12*e2) / det_g
        #   g^2 = (g11*e2 - g12*e1) / det_g
        #
        # But for the dot product grad_s L_i . grad_s L_j, we only need:
        #   grad_s L_i . grad_s L_j = sum_ab (dL_i/d_xa) * g^{ab} * (dL_j/d_xb)
        # where g^{ab} is the inverse metric tensor

        gi11 = g22 * inv_det
        gi12 = -g12 * inv_det
        gi22 = g11 * inv_det

        # Parametric derivatives of barycentric coords: dL_i/dxi, dL_i/deta
        # L1 = 1 - xi - eta  -> (-1, -1)
        # L2 = xi             -> ( 1,  0)
        # L3 = eta            -> ( 0,  1)
        dLdxi = np.empty(3, dtype=np.float64)
        dLdeta = np.empty(3, dtype=np.float64)
        dLdxi[0] = -1.0
        dLdeta[0] = -1.0
        dLdxi[1] = 1.0
        dLdeta[1] = 0.0
        dLdxi[2] = 0.0
        dLdeta[2] = 1.0

        # Local edge map
        lem0 = local_edge_maps[2 * idx, :]
        lem1 = local_edge_maps[2 * idx + 1, :]

        # Field DOFs
        field_ids = tri_to_field[:, itri]

        # Quadrature: integrate grad_s N_i . grad_s N_j over triangle
        K_local = np.zeros((6, 6), dtype=np.float64)

        for iq in range(nq):
            L1 = TRI_DPTS[1, iq]
            L2 = TRI_DPTS[2, iq]
            L3 = TRI_DPTS[3, iq]
            w = weights[iq]

            Ls = np.empty(3, dtype=np.float64)
            Ls[0] = L1
            Ls[1] = L2
            Ls[2] = L3

            # Surface gradients of all 6 P2 basis functions
            # Each has 2 parametric components (dN/dxi, dN/deta)
            dNdxi = np.empty(6, dtype=np.float64)
            dNdeta = np.empty(6, dtype=np.float64)

            # Vertex basis: N_i = L_i(2L_i - 1)
            # dN_i/dxi = (4L_i - 1) * dL_i/dxi
            for iv in range(3):
                q = 4.0 * Ls[iv] - 1.0
                dNdxi[iv] = q * dLdxi[iv]
                dNdeta[iv] = q * dLdeta[iv]

            # Edge basis: N_ij = 4 * L_i * L_j
            # dN_ij/dxi = 4 * (L_j * dL_i/dxi + L_i * dL_j/dxi)
            for ie in range(3):
                ei = lem0[ie]
                ej = lem1[ie]
                dNdxi[3 + ie] = 4.0 * (Ls[ej] * dLdxi[ei] + Ls[ei] * dLdxi[ej])
                dNdeta[3 + ie] = 4.0 * (Ls[ej] * dLdeta[ei] + Ls[ei] * dLdeta[ej])

            # K_ij += w * (grad_s N_i . grad_s N_j)
            # grad_s N_i . grad_s N_j = dNi_xi*g^11*dNj_xi + dNi_xi*g^12*dNj_eta
            #                          + dNi_eta*g^12*dNj_xi + dNi_eta*g^22*dNj_eta
            for i in range(6):
                for j in range(6):
                    K_local[i, j] += w * (
                        dNdxi[i] * gi11 * dNdxi[j]
                        + dNdxi[i] * gi12 * dNdeta[j]
                        + dNdeta[i] * gi12 * dNdxi[j]
                        + dNdeta[i] * gi22 * dNdeta[j]
                    )

        # Scale: weights sum to 1/2 (ref triangle area)
        # Physical integral = 2 * area * ref_integral * kappa_t
        # But we also need the Jacobian: the parametric gradient uses
        # the inverse metric which already accounts for the mapping,
        # so the area factor from the integration measure is just 2*area
        scale = kappa_t * 2.0 * area

        # Write COO entries
        for i in range(6):
            for j in range(6):
                k = p + 6 * i + j
                rows[k] = field_ids[i]
                cols[k] = field_ids[j]
                values[k] = K_local[i, j] * scale

    return values, rows, cols


def assemble_conductive_sheet(field, face_tags, kappa_t):
    """Assemble thin conductive sheet stiffness matrix contribution.

    Models lateral heat conduction through a thin metallic layer
    (e.g. copper foil, ground plane) without volume meshing.

    Args:
        field: Legrange2 field
        mesh: Mesh3D
        face_tags: GMSH face tags for the sheet surface
        kappa: thermal conductivity of the sheet material [W/(m·K)]
        thickness: sheet thickness [m]

    Returns:
        K_values: COO values for stiffness matrix addition
        K_rows: COO row indices
        K_cols: COO column indices
    """
    mesh = field.mesh
    tri_ids = mesh.get_triangles(face_tags)
    n_sel = len(tri_ids)
    nnodes = field.nnodes

    # Precompute local edge maps
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

    return _conductive_sheet_builder(
        mesh.nodes,
        mesh.tris,
        mesh.edges,
        field.tri_to_field,
        tri_ids_2d,
        local_edge_maps,
        kappa_t,
        field.n_field,
    )
