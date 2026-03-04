import warp as wp
import numpy as np

from ...sim.model import Model
from ...sim.state import State
from ...sim.control import Control
from ...sim.contacts import Contacts
from .kernels import (
    advect_vel_kernel,
    apply_forces_kernel,
    compute_divergence_kernel,
    jacobi_pressure_kernel,
    subtract_gradient_kernel
)

class SolverStableFluids:
    """
    Eulerian grid-based fluid solver using the Stable Fluids (Jos Stam) approach.
    """

    def __init__(self, model: Model, pressure_iters: int = 40):
        self.model = model
        self.pressure_iters = pressure_iters
        self.device = model.device

        # 如果场景中存在流体网格，我们需要分配临时缓冲区用于投影步
        if self.model.grid_count > 0:
            self.div_buffer = wp.zeros(self.model.grid_cell_count, dtype=wp.float32, device=self.device)
            self.p_tmp_buffer = wp.zeros(self.model.grid_cell_count, dtype=wp.float32, device=self.device)

    def step(self, state0: State, state1: State, control: Control, contacts: Contacts, dt: float):
        """
        向前推进一个时间步。
        Newton 约定: state0 是当前帧(t)，state1 应该输出为下一帧(t+dt)。
        """
        if self.model.grid_count == 0:
            return

        with wp.ScopedTimer("Stable Fluids Step", active=False):
            # 将 state0 的数据复制到 state1，方便我们在 state1 上进行原地修改
            wp.copy(state1.grid_vel, state0.grid_vel)
            wp.copy(state1.grid_pressure, state0.grid_pressure)
            if state1.grid_density is not None:
                wp.copy(state1.grid_density, state0.grid_density)

            # Newton 支持多网格(比如并行的 RL 环境)，这里循环遍历每个网格发起计算
            for grid_id in range(self.model.grid_count):
                dim_i = self.model.grid_dim.numpy()[grid_id]  # 读取到 CPU 端供维度使用
                dim = wp.vec3i(int(dim_i[0]), int(dim_i[1]), int(dim_i[2]))
                dx = float(self.model.grid_dx.numpy()[grid_id])
                offset = int(self.model.grid_cell_start.numpy()[grid_id])
                
                launch_dim = (int(dim[0]), int(dim[1]), int(dim[2]))
                
                # 读取该网格所处 world 的重力
                world_idx = int(self.model.grid_world.numpy()[grid_id])
                world_idx = max(0, world_idx) # 全局(-1)使用0号world重力
                gravity = self.model.gravity.numpy()[world_idx]
                gravity_wp = wp.vec3(gravity[0], gravity[1], gravity[2])

                # -------------------------------------------------------------
                # 1. 外力 (Add Forces)
                # -------------------------------------------------------------
                wp.launch(
                    kernel=apply_forces_kernel,
                    dim=launch_dim,
                    inputs=[state1.grid_vel, gravity_wp, dt, offset, dim],
                    device=self.device
                )

                # -------------------------------------------------------------
                # 2. 速度平流 (Advection) - 需要将 state1 的当前状态存入 prev
                # -------------------------------------------------------------
                wp.copy(state1.grid_vel_prev, state1.grid_vel)
                wp.launch(
                    kernel=advect_vel_kernel,
                    dim=launch_dim,
                    inputs=[state1.grid_vel_prev, state1.grid_vel, dim, dx, dt, offset],
                    device=self.device
                )

                # 如果有密度场，同样执行平流 (你可以仿照 advect_vel 写一个 advect_scalar_kernel)
                # if state1.grid_density is not None: ...

                # -------------------------------------------------------------
                # 3. 投影 - 散度计算 (Divergence)
                # -------------------------------------------------------------
                wp.launch(
                    kernel=compute_divergence_kernel,
                    dim=launch_dim,
                    inputs=[state1.grid_vel, self.div_buffer, dim, dx, offset],
                    device=self.device
                )

                # -------------------------------------------------------------
                # 4. 投影 - 泊松压力求解 (Pressure Solve via Jacobi)
                # -------------------------------------------------------------
                # 压力初值通常可复用上一帧以加速收敛，这里使用 state1.grid_pressure
                for _ in range(self.pressure_iters):
                    wp.launch(
                        kernel=jacobi_pressure_kernel,
                        dim=launch_dim,
                        inputs=[state1.grid_pressure, self.p_tmp_buffer, self.div_buffer, dim, dx, offset],
                        device=self.device
                    )
                    # Ping-pong swap
                    state1.grid_pressure, self.p_tmp_buffer = self.p_tmp_buffer, state1.grid_pressure

                # -------------------------------------------------------------
                # 5. 投影 - 速度修正 (Subtract Gradient)
                # -------------------------------------------------------------
                wp.launch(
                    kernel=subtract_gradient_kernel,
                    dim=launch_dim,
                    inputs=[state1.grid_vel, state1.grid_pressure, dim, dx, offset],
                    device=self.device
                )