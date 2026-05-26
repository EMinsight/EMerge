import numpy as np
from numba import njit, f8, i8, types, prange
from .heatflux import TRI_DPTS, _local_tri_edge_map, assemble_surface_flux
from ....elements.leg2 import Legrange2

KC = 5.670374419e-8

############################################################
#              TRIANGLE SURFACE MASS ASSEMBLY             #
############################################################


@njit(f8[:, :](i8[:, :]), cache=True, nogil=True)
def _tri_surface_mass_element(local_edge_map):
    weights = TRI_DPTS[0, :]
    nq = weights.shape[0]

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
            li = Ls[local_edge_map[0, ie]]
            lj = Ls[local_edge_map[1, ie]]
            N[3 + ie] = 4.0 * li * lj

        for i in range(6):
            for j in range(6):
                M[i, j] += w * N[i] * N[j]

    return M


############################################################
#                OPTIMIZED ASSEMBLER ROUTINE               #
############################################################


@njit(
    types.Tuple((f8[:], i8[:], i8[:]))(
        f8[:, :], i8[:, :], i8[:, :], i8[:, :], i8[:, :], f8
    ),
    cache=True,
    nogil=True,
    parallel=True,
)
def _robin_stiffness_builder(
    nodes, tris, tri_to_field, tri_ids_2d, local_edge_maps, h_coeff
):
    n_sel = tri_ids_2d.shape[1]
    nnz = n_sel * 36

    values = np.empty(nnz, dtype=np.float64)
    rows = np.empty(nnz, dtype=np.int64)
    cols = np.empty(nnz, dtype=np.int64)

    for idx in prange(n_sel):
        p = idx * 36
        itri = tri_ids_2d[0, idx]

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

        lem0 = local_edge_maps[2 * idx, :]
        lem1 = local_edge_maps[2 * idx + 1, :]
        lem = np.empty((2, 3), dtype=np.int64)
        lem[0, :] = lem0
        lem[1, :] = lem1

        M = _tri_surface_mass_element(lem)

        scale = h_coeff * 2.0 * area

        field_ids = tri_to_field[:, itri]

        for i in range(6):
            for j in range(6):
                k = p + 6 * i + j
                rows[k] = field_ids[i]
                cols[k] = field_ids[j]
                values[k] = M[i, j] * scale

    return values, rows, cols


@njit(
    types.Tuple((f8[:], i8[:], i8[:], f8[:]))(
        f8[:, :], i8[:, :], i8[:, :], i8[:, :], i8[:, :], f8[:], f8, f8, i8
    ),
    cache=True,
    nogil=True,
    parallel=False,
)
def _radiation_builder(
    nodes,
    tris,
    tri_to_field,
    tri_ids_2d,
    local_edge_maps,
    T_dofs,
    emissivity,
    T_amb,
    n_field,
):
    n_triangles = tri_ids_2d.shape[1]
    nnz = n_triangles * 36
    sigma = KC

    K_values = np.empty(nnz, dtype=np.float64)
    K_rows = np.empty(nnz, dtype=np.int64)
    K_cols = np.empty(nnz, dtype=np.int64)
    f_global = np.zeros(n_field, dtype=np.float64)

    weights = TRI_DPTS[0, :]
    nq = weights.shape[0]
    T_amb2 = T_amb * T_amb

    for i_triangle in range(n_triangles):
        p = i_triangle * 36
        itri = tri_ids_2d[0, i_triangle]

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

        lem0 = local_edge_maps[2 * i_triangle, :]
        lem1 = local_edge_maps[2 * i_triangle + 1, :]

        field_ids = tri_to_field[:, itri]
        T_local = np.empty(6, dtype=np.float64)
        for i in range(6):
            T_local[i] = T_dofs[field_ids[i]]

        K_local = np.zeros((6, 6), dtype=np.float64)
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

            N = np.empty(6, dtype=np.float64)
            for iv in range(3):
                N[iv] = Ls[iv] * (2.0 * Ls[iv] - 1.0)
            for ie in range(3):
                li = Ls[lem0[ie]]
                lj = Ls[lem1[ie]]
                N[3 + ie] = 4.0 * li * lj

            T_q = 0.0
            for i in range(6):
                T_q += N[i] * T_local[i]

            T_q2 = T_q * T_q
            h_rad = emissivity * sigma * (T_q2 + T_amb2) * (T_q + T_amb)
            wh = w * h_rad

            for i in range(6):
                for j in range(6):
                    K_local[i, j] += wh * N[i] * N[j]

            wht = wh * T_amb
            for i in range(6):
                f_local[i] += wht * N[i]

        scale = 2.0 * area

        for i in range(6):
            for j in range(6):
                k = p + 6 * i + j
                K_rows[k] = field_ids[i]
                K_cols[k] = field_ids[j]
                K_values[k] = K_local[i, j] * scale

        for i in range(6):
            f_global[field_ids[i]] += f_local[i] * scale

    return K_values, K_rows, K_cols, f_global


############################################################
#                 PYTHON ASSEMBLER INTERFACE               #
############################################################


def assemble_radiation_bc(
    field: Legrange2,
    face_tags: list[int],
    emissivity: float,
    T_amb: float,
    T_solution: np.ndarray,
):
    """Assemble blackbody radiation boundary condition.

    Linearized as Robin BC with temperature-dependent coefficient:
        h_rad = eps * sigma * (T^2 + T_amb^2) * (T + T_amb)

    Args:
        field: Legrange2 field
        face_tags: GMSH face tags for radiation boundary
        emissivity: surface emissivity (0 to 1)
        T_amb: ambient radiation temperature [K]
        T_solution: current temperature solution vector (n_field,)
                    used to evaluate h_rad at quadrature points

    Returns:
        K_values, K_rows, K_cols: COO stiffness contribution
        f_robin: (n_field,) load vector contribution
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

    # Ensure T_solution is the right type
    T_dofs = np.asarray(T_solution, dtype=np.float64)

    return _radiation_builder(
        mesh.nodes,
        mesh.tris,
        field.tri_to_field,
        tri_ids_2d,
        local_edge_maps,
        T_dofs,
        float(emissivity),
        float(T_amb),
        field.n_field,
    )


def assemble_robin_bc(
    field: Legrange2, face_tags: list[int], h_coeff: float, T_amb: float
):
    """Assembles the Robin boundary condition which is the Convection boundary condition to an ambient temperature.

    Args:
        field (Legrange2): The problems Legrange basis function field object
        face_tags (list[int]): A list of face tag integers
        h_coeff (float): The heat flux coefficient in W/m^2
        T_amb (float): The ambient temperature
    """
    mesh = field.mesh
    tri_ids = mesh.get_triangles(face_tags)
    n_sel = len(tri_ids)
    nnodes = field.nnodes

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

    K_values, K_rows, K_cols = _robin_stiffness_builder(
        mesh.nodes,
        mesh.tris,
        field.tri_to_field,
        tri_ids_2d,
        local_edge_maps,
        float(h_coeff),
    )

    f_robin = assemble_surface_flux(field, mesh, face_tags, float(h_coeff * T_amb))

    return K_values, K_rows, K_cols, f_robin
