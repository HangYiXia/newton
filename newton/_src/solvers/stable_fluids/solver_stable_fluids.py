# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import warp as wp

from newton._src.sim.model import Model
from newton._src.sim.state import State
from newton._src.solvers.solver import SolverBase


# --------------------------------------------------------------------------- #
# Warp Kernels
# --------------------------------------------------------------------------- #

@wp.func
def cyclic_index(idx: int, n: int):
    """Handles periodic boundary conditions."""
    ret_idx = idx % n
    if ret_idx < 0:
        ret_idx += n
    return ret_idx


@wp.kernel
def advect_scalar(
    dt: float,
    velocity: wp.array2d(dtype=wp.vec2),
    quantity_in: wp.array2d(dtype=float),
    quantity_out: wp.array2d(dtype=float),
    n: int,
    dh: float,
):
    """Advects a scalar field (e.g. density) using the velocity field."""
    i, j = wp.tid()

    # Semi-Lagrangian backtrace
    # Position in grid units: (i, j)
    # Velocity is physical (units/sec), convert to grid units: v / dh
    vel = velocity[i, j]
    center_x = wp.float32(i) - vel[0] * dt / dh
    center_y = wp.float32(j) - vel[1] * dt / dh

    # Compute indices of source cells
    left_idx = wp.int32(wp.floor(center_x))
    bot_idx = wp.int32(wp.floor(center_y))

    # Bilinear interpolation weights
    s1 = center_x - wp.float32(left_idx)
    s0 = 1.0 - s1
    t1 = center_y - wp.float32(bot_idx)
    t0 = 1.0 - t1

    # Wrap indices
    i0 = cyclic_index(left_idx, n)
    i1 = cyclic_index(left_idx + 1, n)
    j0 = cyclic_index(bot_idx, n)
    j1 = cyclic_index(bot_idx + 1, n)

    quantity_out[i, j] = s0 * (t0 * quantity_in[i0, j0] + t1 * quantity_in[i0, j1]) + \
                         s1 * (t0 * quantity_in[i1, j0] + t1 * quantity_in[i1, j1])


@wp.kernel
def advect_vector(
    dt: float,
    velocity_field: wp.array2d(dtype=wp.vec2),
    quantity_in: wp.array2d(dtype=wp.vec2),
    quantity_out: wp.array2d(dtype=wp.vec2),
    n: int,
    dh: float,
):
    """Advects a vector field (e.g. velocity itself) using the velocity field."""
    i, j = wp.tid()

    vel = velocity_field[i, j]
    center_x = wp.float32(i) - vel[0] * dt / dh
    center_y = wp.float32(j) - vel[1] * dt / dh

    left_idx = wp.int32(wp.floor(center_x))
    bot_idx = wp.int32(wp.floor(center_y))

    s1 = center_x - wp.float32(left_idx)
    s0 = 1.0 - s1
    t1 = center_y - wp.float32(bot_idx)
    t0 = 1.0 - t1

    i0 = cyclic_index(left_idx, n)
    i1 = cyclic_index(left_idx + 1, n)
    j0 = cyclic_index(bot_idx, n)
    j1 = cyclic_index(bot_idx + 1, n)

    quantity_out[i, j] = s0 * (t0 * quantity_in[i0, j0] + t1 * quantity_in[i0, j1]) + \
                         s1 * (t0 * quantity_in[i1, j0] + t1 * quantity_in[i1, j1])


@wp.kernel
def divergence(
    velocity: wp.array2d(dtype=wp.vec2),
    div_out: wp.array2d(dtype=float),
    n: int,
    dh: float,
):
    """Computes the divergence of the velocity field using central differencing."""
    i, j = wp.tid()

    v_right = velocity[cyclic_index(i + 1, n), j][0]
    v_left = velocity[cyclic_index(i - 1, n), j][0]
    v_up = velocity[i, cyclic_index(j + 1, n)][1]
    v_down = velocity[i, cyclic_index(j - 1, n)][1]

    div_out[i, j] = 0.5 * (v_right - v_left + v_up - v_down) / dh


@wp.kernel
def jacobi_iter(
    div_in: wp.array2d(dtype=float),
    p_in: wp.array2d(dtype=float),
    p_out: wp.array2d(dtype=float),
    n: int,
    dh: float,
):
    """Performs one Jacobi iteration to solve the Poisson equation for pressure."""
    i, j = wp.tid()

    p_left = p_in[cyclic_index(i - 1, n), j]
    p_right = p_in[cyclic_index(i + 1, n), j]
    p_down = p_in[i, cyclic_index(j - 1, n)]
    p_up = p_in[i, cyclic_index(j + 1, n)]

    # Standard Poisson update on a grid with spacing dh
    p_out[i, j] = 0.25 * (p_left + p_right + p_down + p_up - dh * dh * div_in[i, j])


@wp.kernel
def subtract_gradient(
    pressure: wp.array2d(dtype=float),
    velocity: wp.array2d(dtype=wp.vec2),
    n: int,
    dh: float,
):
    """Projects the velocity field to be incompressible by subtracting the pressure gradient."""
    i, j = wp.tid()

    p_right = pressure[cyclic_index(i + 1, n), j]
    p_left = pressure[cyclic_index(i - 1, n), j]
    p_up = pressure[i, cyclic_index(j + 1, n)]
    p_down = pressure[i, cyclic_index(j - 1, n)]

    grad_x = 0.5 * (p_right - p_left) / dh
    grad_y = 0.5 * (p_up - p_down) / dh

    v = velocity[i, j]
    velocity[i, j] = wp.vec2(v[0] - grad_x, v[1] - grad_y)


# --------------------------------------------------------------------------- #
# Solver Class
# --------------------------------------------------------------------------- #

class SolverStableFluids(SolverBase):
    """
    A grid-based fluid solver implementing Jos Stam's 'Stable Fluids' algorithm.
    Includes semi-Lagrangian advection and a pressure projection step for incompressibility.
    """

    def __init__(self, model: Model, pressure_iterations: int = 25):
        super().__init__(model)
        
        if model.fluid_res <= 0:
            raise ValueError(
                "Model does not have a fluid grid configured. "
                "Please call `model.add_fluid_grid(res, cell_size)` in the builder."
            )

        self.res = model.fluid_res
        self.dh = model.fluid_cell_size
        self.pressure_iterations = pressure_iterations

        # Allocate temporary buffers
        shape = (self.res, self.res)
        self.div_array = wp.zeros(shape, dtype=float, device=model.device)
        self.p_aux = wp.zeros(shape, dtype=float, device=model.device)

    def step(
        self,
        state_in: State,
        state_out: State,
        control=None,
        contacts=None,
        dt: float = 1.0 / 60.0
    ):
        """
        Advances the fluid simulation by one time step.
        """
        
        # 1. Advect Velocity (Self-Advection)
        #    Input: state_in.velocity
        #    Output: state_out.velocity (Intermediate 'w' field)
        wp.launch(
            advect_vector,
            dim=(self.res, self.res),
            inputs=[
                dt,
                state_in.fluid_velocity,    # Velocity field used for transport
                state_in.fluid_velocity,    # Quantity being transported (velocity itself)
                state_out.fluid_velocity,   # Result
                self.res,
                self.dh
            ]
        )

        # 2. Projection Step (Make Velocity Incompressible)
        #    Compute Divergence of the intermediate velocity field
        wp.launch(
            divergence,
            dim=(self.res, self.res),
            inputs=[state_out.fluid_velocity, self.div_array, self.res, self.dh]
        )

        #    Solve Pressure Poisson Equation (Jacobi Iteration)
        #    Initialize output pressure with the previous step's pressure (Warm Start)
        wp.copy(state_out.fluid_pressure, state_in.fluid_pressure)
        
        #    Ping-pong pointers for iteration
        p_src = state_out.fluid_pressure
        p_dst = self.p_aux

        for _ in range(self.pressure_iterations):
            wp.launch(
                jacobi_iter,
                dim=(self.res, self.res),
                inputs=[self.div_array, p_src, p_dst, self.res, self.dh]
            )
            # Swap buffers
            p_src, p_dst = p_dst, p_src

        #    Ensure final result is in state_out.fluid_pressure
        if p_src.ptr != state_out.fluid_pressure.ptr:
            wp.copy(state_out.fluid_pressure, p_src)

        #    Update Velocity: u = w - grad(p)
        #    (Updates state_out.fluid_velocity in-place)
        wp.launch(
            subtract_gradient,
            dim=(self.res, self.res),
            inputs=[state_out.fluid_pressure, state_out.fluid_velocity, self.res, self.dh]
        )

        # 3. Advect Density
        #    Input: state_in.density
        #    Transported by: state_out.velocity (the new incompressible field)
        #    Output: state_out.density
        wp.launch(
            advect_scalar,
            dim=(self.res, self.res),
            inputs=[
                dt,
                state_out.fluid_velocity,
                state_in.fluid_density,
                state_out.fluid_density,
                self.res,
                self.dh
            ]
        )