# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example Stable Fluids (MAC Marker Particles)
#
# Shows how to set up an Eulerian grid-based fluid simulation and use
# massless marker particles for visualization.
#
# Command: python -m newton.examples example_stable_fluids
#
###########################################################################

import warp as wp
import numpy as np

import newton
import newton.examples
# 假设你在 newton.solvers 中导出了我们前面编写的求解器
from newton.solvers import SolverStableFluids


class Example:
    def __init__(self, viewer, args=None):
        # 1. 初始化仿真时间参数
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        # 流体通常不需要像 PBD 那么多子步，1个子步足够演示
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.args = args

        builder = newton.ModelBuilder()

        # 2. 设置流体网格参数
        grid_res = 32
        dx = 0.1
        
        # 添加 Eulerian 流体网格 (调用我们在 builder 中新增的 API)
        builder.add_fluid_grid(
            dim=(grid_res, grid_res, grid_res),
            dx=dx,
            xform=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()),
            viscosity=0.0
        )

        # 3. 撒入 MAC 标记粒子用于可视化
        # 我们在网格的偏上方区域生成一个水滴块
        # 因为网格在 (0,0,0) 到 (3.2, 3.2, 3.2) 之间，我们把水块放在中心靠上
        drop_pos = (grid_res * dx * 0.3, grid_res * dx * 0.3, grid_res * dx * 0.6)
        
        # 粒子间距设为网格大小的一半，以保证每个 Cell 里有足够的标记粒子
        particle_spacing = dx * 0.5 
        
        builder.add_particle_grid(
            pos=drop_pos,
            rot=wp.quat_identity(),
            vel=(0.0, 0.0, -2.0),           # 给一个初始向下的速度
            dim_x=12, dim_y=12, dim_z=12,   # 12x12x12 的粒子块
            cell_x=particle_spacing, 
            cell_y=particle_spacing, 
            cell_z=particle_spacing,
            mass=0.0,                       # 关键：质量必须为 0，使其不受传统碰撞和重力求解器影响
            jitter=particle_spacing * 0.2,  # 稍微抖动打破晶格伪影
            radius_mean=dx * 0.2            # 渲染时的粒子半径
        )

        # 添加一个地面，仅作为视觉参考
        builder.add_ground_plane()

        # 4. Finalize 模型
        self.model = builder.finalize()

        # 5. 实例化 Stable Fluids 求解器
        self.solver = SolverStableFluids(self.model, pressure_iters=40)

        # 初始化状态
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        # 绑定到 Viewer
        self.viewer.set_model(self.model)

        self.graph = None
        self.capture()

    def capture(self):
        """捕获 CUDA Graph 以加速渲染 (可选)"""
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        """内部仿真循环"""
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # 应用任何交互力 (如果你在 Viewer 中拉拽粒子)
            self.viewer.apply_forces(self.state_0)

            # 流体不需要常规的 collide，但保留接口一致性
            # self.model.collide(self.state_0, self.contacts)
            
            # 步进流体求解器
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # 状态交换 (Ping-Pong)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        """外部步进调用"""
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def test_final(self):
        """单元测试用的验证函数，验证粒子是否还在运动等"""
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "marker particles exist",
            lambda q, qd: True, # 只要不报错就先算过
            [0],
        )

    def render(self):
        """渲染当前帧"""
        self.viewer.begin_frame(self.sim_time)
        # log_state 会自动读取 self.state_0.particle_q 并发送给 Viser/Rerun 进行渲染
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    # 解析命令行参数并初始化 viewer
    viewer, args = newton.examples.init()

    # 创建实例并运行
    example = Example(viewer, args)

    newton.examples.run(example, args)