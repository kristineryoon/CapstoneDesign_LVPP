"""
Solve obstacle problem with Proximal Galerkin, Galahad, and IPOPT and compare the results

Author: Jørgen S. Dokken
SPDX-License-Identifier: MIT
"""

import argparse
from pathlib import Path

import dolfinx
import numpy as np
from obstacle_ipopt_galahad import ObstacleProblem, setup_problem
from obstacle_pg import solve_problem
from obstacle_snes import snes_solve

from lvpp.optimization import galahad_solver, ipopt_solver

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Solve the obstacle problem on a unit square using Galahad.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--path",
        "-P",
        dest="infile",
        type=Path,
        default="./meshes/disk_3.xdmf",
        help="Path to infile",
    )
    parser.add_argument(
        "--results",
        "-O",
        dest="result_dir",
        type=Path,
        default=Path("results"),
        help="Path to results ",
    )
    max_iter = 500
    tol = 1e-4
    args = parser.parse_args()

    args.result_dir.mkdir(parents=True, exist_ok=True)

    # Set up problem matrices. initial guess and bounds
    problem = setup_problem(args.infile)
    S_, M_, f_, bounds_ = setup_problem(args.infile)

    bounds = tuple(b.x.array for b in bounds_)

    problem = ObstacleProblem(S_.copy(), M_.copy(), f_.x.array)
    V = f_.function_space
    mesh = V.mesh
    degree = mesh.geometry.cmap.degree
    V_out = dolfinx.fem.functionspace(mesh, ("Lagrange", degree))

    # Solve with Galahad
    x_g = dolfinx.fem.Function(V, name="galahad")
    x_g.x.array[:] = 0.0

    x_galahad, num_galahad_iterations = galahad_solver(
        problem,
        x_g.x.array.copy(),
        bounds,
        max_iter=max_iter,
        use_hessian=True,
        tol=tol,
    )
    x_g.x.array[:] = x_galahad
    x_g_out = dolfinx.fem.Function(V_out, name="ipopt")
    x_g_out.interpolate(x_g)
    with dolfinx.io.VTXWriter(
        V.mesh.comm, args.result_dir / f"{args.infile.stem}_galahad.bp", [x_g_out]
    ) as bp:
        bp.write(0.0)

       # Solve with llvp (first order)

    u_lvpp, max_it = solve_problem(
        args.infile,
        1,
        maximum_number_of_outer_loop_iterations=max_iter,
        alpha_scheme="double_exponential",
        alpha_max=1e2,
        tol_exit=tol,
    )

    mesh = u_lvpp.function_space.mesh
    degree = mesh.geometry.cmap.degree
    V_out = dolfinx.fem.functionspace(mesh, ("Lagrange", degree))

    # Membrane displacement u
    u_out = dolfinx.fem.Function(V_out, name="llvp")
    u_out.interpolate(u_lvpp.sub(0))

    # Obstacle function phi on the same function space as u_out
    def phi_set_for_output(x):
        phi_vals = np.full_like(x[0], -10.0)

        R = 0.1
        L = 0.4
        Z_top = -0.2
        Z_center = Z_top - R

        dx = np.abs(x[0] - np.round(x[0] / L) * L)
        dy = np.abs(x[1] - np.round(x[1] / L) * L)

        mask_y_pipe = dx <= R
        mask_x_pipe = dy <= R

        h_y_pipe = np.full_like(x[0], -10.0)
        h_y_pipe[mask_y_pipe] = Z_center + np.sqrt(R**2 - dx[mask_y_pipe]**2)

        h_x_pipe = np.full_like(x[0], -10.0)
        h_x_pipe[mask_x_pipe] = Z_center + np.sqrt(R**2 - dy[mask_x_pipe]**2)

        phi_vals = np.maximum(phi_vals, h_y_pipe)
        phi_vals = np.maximum(phi_vals, h_x_pipe)

        return phi_vals

    phi_out = dolfinx.fem.Function(V_out, name="phi")
    phi_out.interpolate(phi_set_for_output)

    # Gap function: gap = u - phi
    gap_out = dolfinx.fem.Function(V_out, name="gap")
    gap_out.x.array[:] = u_out.x.array[:] - phi_out.x.array[:]

    # Contact indicator: contact = 1 if gap <= eps, otherwise 0
    eps = 1e-5
    contact_out = dolfinx.fem.Function(V_out, name="contact")
    contact_out.x.array[:] = (gap_out.x.array[:] <= eps).astype(gap_out.x.array.dtype)

    # Save llvp, phi, gap, contact together
    with dolfinx.io.VTXWriter(
        mesh.comm,
        args.result_dir / f"{args.infile.stem}_llvp_first_order_contact.bp",
        [u_out, phi_out, gap_out, contact_out],
    ) as bp:
        bp.write(0.0)

    # Solve with llvp (second order)

    u_lvpp_2, max_it_2 = solve_problem(
        args.infile,
        2,
        maximum_number_of_outer_loop_iterations=max_iter,
        alpha_scheme="double_exponential",
        alpha_max=1e2,
        tol_exit=tol,
    )

    # Membrane displacement u
    u_out = u_lvpp_2.sub(0).collapse()
    u_out.name = "llvp"

    V_out_2 = u_out.function_space
    mesh_2 = V_out_2.mesh

    # Obstacle function phi on the same function space as u_out
    phi_out = dolfinx.fem.Function(V_out_2, name="phi")
    phi_out.interpolate(phi_set_for_output)

    # Gap function: gap = u - phi
    gap_out = dolfinx.fem.Function(V_out_2, name="gap")
    gap_out.x.array[:] = u_out.x.array[:] - phi_out.x.array[:]

    # Contact indicator
    eps = 1e-5
    contact_out = dolfinx.fem.Function(V_out_2, name="contact")
    contact_out.x.array[:] = (gap_out.x.array[:] <= eps).astype(gap_out.x.array.dtype)

    # Save llvp, phi, gap, contact together
    with dolfinx.io.VTXWriter(
        mesh_2.comm,
        args.result_dir / f"{args.infile.stem}_llvp_second_order_contact.bp",
        [u_out, phi_out, gap_out, contact_out],
    ) as bp:
        bp.write(0.0)

    # Solve with IPOPT (With hessian)
    ipopt_iteration_count = {}
    for with_hessian in [True, False]:
        x_i = dolfinx.fem.Function(V, name="ipopt")
        x_i.x.array[:] = 0.0
        x_ipopt = ipopt_solver(
            problem,
            x_i.x.array.copy(),
            bounds,
            max_iter=max_iter,
            tol=1e-2 * tol,
            activate_hessian=with_hessian,
        )
        ipopt_iteration_count[with_hessian] = problem.total_iteration_count
        x_i.x.array[:] = x_ipopt

        # Output on geometry space

        x_i_out = dolfinx.fem.Function(V_out, name="ipopt")
        x_i_out.interpolate(x_i)
        with dolfinx.io.VTXWriter(
            mesh.comm,
            args.result_dir / f"{args.infile.stem}_ipopt_hessian_{with_hessian}.bp",
            [x_i_out],
        ) as bp:
            bp.write(0.0)

    # Solve with SNES
    u_snes, num_snes_iterations = snes_solve(
        args.infile,
        snes_options={
            "snes_type": "vinewtonssls",
            "snes_monitor": None,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "snes_error_if_not_converged": True,
            "ksp_error_if_not_converged": True,
        },
    )
    u_snes.name = "snes"
    with dolfinx.io.VTXWriter(
        mesh.comm,
        args.result_dir / f"{args.infile.stem}_snes.bp",
        [u_snes],
    ) as bp:
        bp.write(0.0)

    print(
        np.min(
            mesh.h(
                mesh.topology.dim, np.arange(mesh.topology.index_map(mesh.topology.dim).size_local)
            )
        )
    )
    print(f"{args.infile} Galahad iterations: {num_galahad_iterations}")
    print(f"{args.infile} llvp iterations: (P=1) {max_it}")
    print(f"{args.infile} llvp iterations: (P=2) {max_it_2}")
    print(f"{args.infile} Ipopt iterations: (With hessian) {ipopt_iteration_count[True]}")
    print(f"{args.infile} Ipopt iterations: (Without hessian {ipopt_iteration_count[False]}")
    print(f"{args.infile} SNES iterations: {num_snes_iterations}")