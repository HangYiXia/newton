# 修改说明

## 架构范式

1. **扁平化与批处理 (Flattening & Batching)**：不使用多维数组（如 3D 数组），而是把所有流体网格的 Cell 摊平为 1D 数组（通过 grid_cell_start 索引），以支持在同一个场景（或多个并行 World）中存在多个独立的流体网格。

2. **状态分离 (Stateless Model)**：静态参数（如网格尺寸、分辨率）放在 Model 中，动态变量（如速度场、密度场）放在 State 中，Python 端通过 ModelBuilder 收集配置并构建 Warp Array。

3. **World 分组 (World Grouping)**：支持并行多环境，需要加入 grid_world_start 逻辑。

# 数据结构

## `newton/_src/sim/state.py`：动态物理量

因实现Stable Fluids添加了

```python
  # --- Fluid Grid State ---
  self.grid_vel: wp.array | None = None
  """网格单元速度场 [m/s], shape (grid_cell_count,), dtype :class:`vec3`."""

  self.grid_vel_prev: wp.array | None = None
  """上一帧的网格单元速度场 (用于平流计算双缓冲), shape (grid_cell_count,), dtype :class:`vec3`."""

  self.grid_density: wp.array | None = None
  """网格单元密度场/染料浓度 [kg/m³ 或无量纲], shape (grid_cell_count,), dtype float."""

  self.grid_density_prev: wp.array | None = None
  """上一帧的网格单元密度场, shape (grid_cell_count,), dtype float."""

  self.grid_pressure: wp.array | None = None
  """网格单元压力场 [Pa], shape (grid_cell_count,), dtype float."""

  self.divergence: wp.array | None = None
  """网格单元速度散度 [1/s], shape (grid_cell_count,), dtype float."""
```

## `newton/_src/sim/model.py`：静态参数配置

因实现Stable Fluids添加了

在原 class AttributeFrequency(IntEnum): 中添加了

```python
  class AttributeFrequency(IntEnum):
      # 新加入：
      GRID = 16
      """Attribute frequency follows the number of grids (see :attr:`~newton.Model.grid_count`)."""
```
```python
  class Model:
      def __init__(self, device: Devicelike | None = None):
          # 新加入：
          # --- Fluid Grid Parameters ---
          self.grid_count = 0
          """Total number of fluid grids in the system."""
          self.grid_cell_count = 0
          """Total number of fluid grid cells in the system (sum of nx*ny*nz for all grids)."""

          self.grid_dim = None
          """网格分辨率 (nx, ny, nz), shape [grid_count], dtype vec3i."""
          self.grid_dx = None
          """网格单元边长 [m], shape [grid_count], dtype float."""
          self.grid_transform = None
          """网格在世界坐标系下的变换, shape [grid_count], dtype transform."""
          self.grid_viscosity = None
          """流体运动粘度, shape [grid_count], dtype float."""
          
          self.grid_cell_start = None
          """每个网格在 1D 展平数组中的起始 cell 索引, shape [grid_count + 1], int."""
          
          self.grid_world = None
          """World index for each grid, shape [grid_count], int. -1 for global."""
          self.grid_world_start = None
          """Start index of the first grid per world, shape [world_count + 2], int."""
          
          self.attribute_frequency["grid_dim"] = Model.AttributeFrequency.GRID
          self.attribute_frequency["grid_dx"] = Model.AttributeFrequency.GRID
          self.attribute_frequency["grid_transform"] = Model.AttributeFrequency.GRID
    def state(self, requires_grad: bool | None = None) -> State:
        # 新加入：
        # fluid grids
        if self.grid_cell_count > 0:
            s.grid_vel = wp.zeros(self.grid_cell_count, dtype=wp.vec3, device=self.device, requires_grad=requires_grad)
            s.grid_vel_prev = wp.zeros(self.grid_cell_count, dtype=wp.vec3, device=self.device, requires_grad=requires_grad)
            s.grid_density = wp.zeros(self.grid_cell_count, dtype=wp.float32, device=self.device, requires_grad=requires_grad)
            s.grid_density_prev = wp.zeros(self.grid_cell_count, dtype=wp.float32, device=self.device, requires_grad=requires_grad)
            s.grid_pressure = wp.zeros(self.grid_cell_count, dtype=wp.float32, device=self.device, requires_grad=requires_grad)
```
## `newton/_src/sim/builder.py`：构建器接口

因实现Stable Fluids添加了

```python
  class ModelBuilder:
      def __init__(self):
          # 新加入：
          # fluid grids
          self.grid_dim = []
          self.grid_dx = []
          self.grid_transform = []
          self.grid_viscosity = []
          self.grid_world = []
          self.grid_cell_start = []
          self.grid_cell_count = 0
          self.grid_world_start = []
      
      # 新加入：
      # 用于创建流体的公有 API
      def add_fluid_grid(
        self,
        dim: tuple[int, int, int],
        dx: float,
        xform: Transform | None = None,
        viscosity: float = 0.0,
        custom_attributes: dict[str, Any] | None = None,
    ) -> int:
        """
        Adds an Eulerian fluid grid to the model for Stable Fluids simulation.

        Args:
            dim: Grid resolution (nx, ny, nz)
            dx: Physical size of a single grid cell
            xform: The world transform of the grid's origin. 
            viscosity: Kinematic viscosity of the fluid.
            custom_attributes: Dictionary of custom attribute names to values.

        Returns:
            The index of the fluid grid in the model.
        """
        if xform is None:
            xform = wp.transform()
        else:
            xform = wp.transform(*xform)
            
        grid_id = len(self.grid_dim)
        
        self.grid_cell_start.append(self.grid_cell_count)
        self.grid_dim.append(wp.vec3i(dim[0], dim[1], dim[2]))
        self.grid_dx.append(dx)
        self.grid_transform.append(xform)
        self.grid_viscosity.append(viscosity)
        self.grid_world.append(self.current_world)
        
        # 累加 cell 的总量
        cells = int(dim[0] * dim[1] * dim[2])
        self.grid_cell_count += cells
        
        if custom_attributes:
            self._process_custom_attributes(
                entity_index=grid_id,
                custom_attrs=custom_attributes,
                expected_frequency=Model.AttributeFrequency.GRID,
            )
            
        return grid_id
        def _build_world_starts(self):
            world_entity_start_arrays = [
                # 新加入：
                (self.grid_world_start, len(self.grid_dim), self.grid_world, "fluid grid"),
            ]
        
        def finalize(self) -> Model:
            with wp.ScopedDevice(device):
                # 新加入：
                # ---------------------
                # fluid grids
                m.grid_count = len(self.grid_dim)
                m.grid_cell_count = self.grid_cell_count
                if m.grid_count > 0:
                    m.grid_dim = wp.array(self.grid_dim, dtype=wp.vec3i, device=device)
                    m.grid_dx = wp.array(self.grid_dx, dtype=wp.float32, device=device)
                    m.grid_transform = wp.array(self.grid_transform, dtype=wp.transform, device=device)
                    m.grid_viscosity = wp.array(self.grid_viscosity, dtype=wp.float32, device=device)
                    m.grid_world = wp.array(self.grid_world, dtype=wp.int32, device=device)
                    m.grid_world_start = wp.array(self.grid_world_start, dtype=wp.int32, device=device)
                    
                    # 补齐 offset 数组的最后一个占位符
                    grid_starts = copy.copy(self.grid_cell_start)
                    grid_starts.append(self.grid_cell_count)
                    m.grid_cell_start = wp.array(grid_starts, dtype=wp.int32, device=device)
```
# Solver实现
以下完全是新增文件：具体说明见文件注释
## `newton/_src/solvers/stable_fluids/__init__.py`：包的导出
## `newton/_src/solvers/__init__.py`：包的导出
## `newton/solvers.py`：newton.solvers 中直接导入的接口

## `newton/_src/solvers/stable_fluids/solver.py`：
在 Newton 中，物理逻辑是由 Solver 的 step() 函数推进的。我们需要在这里组织流体的 Ping-Pong 缓冲和迭代。

## `newton/_src/solvers/stable_fluids/kernels.py`：Stable Fluids 的 Warp 内核实现





[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/newton-physics/newton/main)
[![codecov](https://codecov.io/gh/newton-physics/newton/graph/badge.svg?token=V6ZXNPAWVG)](https://codecov.io/gh/newton-physics/newton)
[![Push - AWS GPU](https://github.com/newton-physics/newton/actions/workflows/push_aws_gpu.yml/badge.svg)](https://github.com/newton-physics/newton/actions/workflows/push_aws_gpu.yml)

# Newton

Newton is a GPU-accelerated physics simulation engine built upon [NVIDIA Warp](https://github.com/NVIDIA/warp), specifically targeting roboticists and simulation researchers.

Newton extends and generalizes Warp's ([deprecated](https://github.com/NVIDIA/warp/discussions/735)) `warp.sim` module, and integrates
[MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) as its primary backend. Newton emphasizes GPU-based computation, [OpenUSD](https://openusd.org/) support, differentiability, and user-defined extensibility, facilitating rapid iteration and scalable robotics simulation.

Newton is a [Linux Foundation](https://www.linuxfoundation.org/) project that is community-built and maintained. Code is licensed under [Apache-2.0](https://github.com/newton-physics/newton/blob/main/LICENSE.md). Documentation is licensed under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).

Newton was initiated by [Disney Research](https://www.disneyresearch.com/), [Google DeepMind](https://deepmind.google/), and [NVIDIA](https://www.nvidia.com/).

## Quickstart

```bash
pip install "newton[examples]"
python -m newton.examples basic_pendulum
```

To install from source with [uv](https://docs.astral.sh/uv/), see the [installation guide](https://newton-physics.github.io/newton/latest/guide/installation.html).


## Examples

Before running the examples below, install Newton with the examples extra:

```bash
pip install "newton[examples]"
```

If you installed from source with uv, substitute `uv run` for `python` in the commands below.

<table>
  <tr>
    <td colspan="3"><h3>Basic Examples</h3></td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/basic/example_basic_pendulum.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_basic_pendulum.jpg" alt="Pendulum">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/basic/example_basic_urdf.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_basic_urdf.jpg" alt="URDF">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/basic/example_basic_viewer.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_basic_viewer.jpg" alt="Viewer">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>python -m newton.examples basic_pendulum</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples basic_urdf</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples basic_viewer</code>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/basic/example_basic_shapes.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_basic_shapes.jpg" alt="Shapes">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/basic/example_basic_joints.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_basic_joints.jpg" alt="Joints">
      </a>
    </td>
    <td align="center" width="33%">
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>python -m newton.examples basic_shapes</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples basic_joints</code>
    </td>
    <td align="center" width="33%">
    </td>
  </tr>
  <tr>
    <td colspan="3"><h3>Robot Examples</h3></td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/robot/example_robot_cartpole.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_robot_cartpole.jpg" alt="Cartpole">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/robot/example_robot_humanoid.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_robot_humanoid.jpg" alt="Humanoid">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/robot/example_robot_g1.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_robot_g1.jpg" alt="G1">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>python -m newton.examples robot_cartpole</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples robot_humanoid</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples robot_g1</code>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/robot/example_robot_h1.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_robot_h1.jpg" alt="H1">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/robot/example_robot_anymal_d.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_robot_anymal_d.jpg" alt="Anymal D">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/robot/example_robot_anymal_c_walk.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_robot_anymal_c_walk.jpg" alt="Anymal C Walk">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>python -m newton.examples robot_h1</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples robot_anymal_d</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples robot_anymal_c_walk</code>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/robot/example_robot_policy.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_robot_policy.jpg" alt="Policy">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/robot/example_robot_ur10.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_robot_ur10.jpg" alt="UR10">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/robot/example_robot_panda_hydro.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_robot_panda_hydro.jpg" alt="Panda Hydro">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>python -m newton.examples robot_policy</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples robot_ur10</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples robot_panda_hydro</code>
    </td>
  </tr>
  <tr>
    <td colspan="3"><h3>Cable Examples</h3></td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/cable/example_cable_bend.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_cable_bend.jpg" alt="Cable Bend">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/cable/example_cable_twist.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_cable_twist.jpg" alt="Cable Twist">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/cable/example_cable_bundle_hysteresis.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_cable_bundle_hysteresis.jpg" alt="Cable Bundle Hysteresis">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>python -m newton.examples cable_bend</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples cable_twist</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples cable_bundle_hysteresis</code>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/cable/example_cable_pile.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_cable_pile.jpg" alt="Cable Pile">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/cable/example_cable_y_junction.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_cable_y_junction.jpg" alt="Cable Y-Junction">
      </a>
    </td>
    <td align="center" width="33%">
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>python -m newton.examples cable_pile</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples cable_y_junction</code>
    </td>
    <td align="center" width="33%">
    </td>
  </tr>
  <tr>
    <td colspan="3"><h3>Cloth Examples</h3></td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/cloth/example_cloth_bending.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_cloth_bending.jpg" alt="Cloth Bending">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/cloth/example_cloth_hanging.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_cloth_hanging.jpg" alt="Cloth Hanging">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/cloth/example_cloth_style3d.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_cloth_style3d.jpg" alt="Cloth Style3D">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>python -m newton.examples cloth_bending</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples cloth_hanging</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples cloth_style3d</code>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/cloth/example_cloth_h1.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_cloth_h1.jpg" alt="Cloth H1">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/cloth/example_cloth_twist.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_cloth_twist.jpg" alt="Cloth Twist">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/cloth/example_cloth_rollers.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_cloth_rolling_cloth.jpg" alt="Cloth Rollers">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>python -m newton.examples cloth_h1</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples cloth_twist</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples cloth_rollers</code>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/cloth/example_cloth_poker_cards.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_multiphysics_poker_cards_stacking.jpg" alt="Cloth Poker Cards">
      </a>
    </td>
    <td align="center" width="33%">
    </td>
    <td align="center" width="33%">
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>python -m newton.examples cloth_poker_cards</code>
    </td>
    <td align="center" width="33%">
    </td>
    <td align="center" width="33%">
    </td>
  </tr>
  <tr>
    <td colspan="3"><h3>Inverse Kinematics Examples</h3></td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/ik/example_ik_franka.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_ik_franka.jpg" alt="IK Franka">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/ik/example_ik_h1.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_ik_h1.jpg" alt="IK H1">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/ik/example_ik_benchmark.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_ik_benchmark.jpg" alt="IK Benchmark">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>python -m newton.examples ik_franka</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples ik_h1</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples ik_benchmark</code>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/ik/example_ik_custom.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_ik_custom.jpg" alt="IK Custom">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/cloth/example_cloth_franka.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_cloth_franka.jpg" alt="Cloth Franka">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/ik/example_ik_cube_stacking.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_ik_cube_stacking.jpg" alt="Stack Cubes">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>python -m newton.examples ik_custom</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples cloth_franka</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples ik_cube_stacking</code>
    </td>
  </tr>
  <tr>
    <td colspan="3"><h3>MPM Examples</h3></td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/mpm/example_mpm_granular.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_mpm_granular.jpg" alt="MPM Granular">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/mpm/example_mpm_anymal.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_mpm_anymal.jpg" alt="MPM Anymal">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/mpm/example_mpm_twoway_coupling.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_mpm_twoway_coupling.jpg" alt="MPM two-way coupling">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>python -m newton.examples mpm_granular</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples mpm_anymal</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples mpm_twoway_coupling</code>
    </td>
  </tr>
  <tr>
    <td colspan="3"><h3>Sensor Examples</h3></td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/sensors/example_sensor_contact.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_sensor_contact.jpg" alt="Sensor Contact">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/sensors/example_sensor_tiled_camera.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_sensor_tiled_camera.jpg" alt="Sensor Tiled Camera">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/sensors/example_sensor_imu.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_sensor_imu.jpg" alt="Sensor IMU">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>python -m newton.examples sensor_contact</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples sensor_tiled_camera</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples sensor_imu</code>
    </td>
  </tr>
  <tr>
    <td colspan="3"><h3>Selection Examples</h3></td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/selection/example_selection_cartpole.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_selection_cartpole.jpg" alt="Selection Cartpole">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/selection/example_selection_materials.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_selection_materials.jpg" alt="Selection Materials">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/selection/example_selection_articulations.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_selection_articulations.jpg" alt="Selection Articulations">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>python -m newton.examples selection_cartpole</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples selection_materials</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples selection_articulations</code>
    </td>
  </tr>
  <tr>
    <td colspan="3"><h3>DiffSim Examples</h3></td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/diffsim/example_diffsim_ball.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_diffsim_ball.jpg" alt="DiffSim Ball">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/diffsim/example_diffsim_cloth.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_diffsim_cloth.jpg" alt="DiffSim Cloth">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/diffsim/example_diffsim_drone.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_diffsim_drone.jpg" alt="DiffSim Drone">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>python -m newton.examples diffsim_ball</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples diffsim_cloth</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples diffsim_drone</code>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/diffsim/example_diffsim_spring_cage.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_diffsim_spring_cage.jpg" alt="DiffSim Spring Cage">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/diffsim/example_diffsim_soft_body.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_diffsim_soft_body.jpg" alt="DiffSim Soft Body">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/diffsim/example_diffsim_bear.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_diffsim_bear.jpg" alt="DiffSim Quadruped">
      </a>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>python -m newton.examples diffsim_spring_cage</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples diffsim_soft_body</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples diffsim_bear</code>
    </td>
  </tr>
  <tr>
    <td colspan="3"><h3>Multi-Physics Examples</h3></td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/multiphysics/example_softbody_gift.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_multiphysics_falling_gift.jpg" alt="Softbody Gift">
      </a>
    </td>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/multiphysics/example_softbody_dropping_to_cloth.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_multiphysics_softbody_dropping_to_cloth.jpg" alt="Softbody Dropping to Cloth">
      </a>
    </td>
    <td align="center" width="33%">
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>python -m newton.examples softbody_gift</code>
    </td>
    <td align="center" width="33%">
      <code>python -m newton.examples softbody_dropping_to_cloth</code>
    </td>
    <td align="center" width="33%">
    </td>
  </tr>
  <tr>
    <td colspan="3"><h3>Softbody Examples</h3></td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="https://github.com/newton-physics/newton/blob/main/newton/examples/softbody/example_softbody_hanging.py">
        <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_softbody_hanging.jpg" alt="Softbody Hanging">
      </a>
    </td>
    <td align="center" width="33%">
    </td>
    <td align="center" width="33%">
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <code>python -m newton.examples softbody_hanging</code>
    </td>
    <td align="center" width="33%">
    </td>
    <td align="center" width="33%">
    </td>
  </tr>
</table>

### Example Options

The examples support the following command-line arguments:

| Argument        | Description                                                                                         | Default                      |
| --------------- | --------------------------------------------------------------------------------------------------- | ---------------------------- |
| `--viewer`      | Viewer type: `gl` (OpenGL window), `usd` (USD file output), `rerun` (ReRun), or `null` (no viewer). | `gl`                         |
| `--device`      | Compute device to use, e.g., `cpu`, `cuda:0`, etc.                                                  | `None` (default Warp device) |
| `--num-frames`  | Number of frames to simulate (for USD output).                                                      | `100`                        |
| `--output-path` | Output path for USD files (required if `--viewer usd` is used).                                     | `None`                       |

Some examples may add additional arguments (see their respective source files for details).

### Example Usage

```bash
# List available examples
python -m newton.examples

# Run with the USD viewer and save to my_output.usd
python -m newton.examples basic_viewer --viewer usd --output-path my_output.usd

# Run on a selected device
python -m newton.examples basic_urdf --device cuda:0

# Combine options
python -m newton.examples basic_viewer --viewer gl --num-frames 500 --device cpu
```

## Contributing and Development

See the [contribution guidelines](https://github.com/newton-physics/newton-governance/blob/main/CONTRIBUTING.md) and the [development guide](https://newton-physics.github.io/newton/latest/guide/development.html) for instructions on how to contribute to Newton.

## Support and Community Discussion

For questions, please consult the [Newton documentation](https://newton-physics.github.io/newton/latest/guide/overview.html) first before creating [a discussion in the main repository](https://github.com/newton-physics/newton/discussions).

## Code of Conduct

By participating in this community, you agree to abide by the Linux Foundation [Code of Conduct](https://lfprojects.org/policies/code-of-conduct/).

## Project Governance, Legal, and Members

Please see the [newton-governance repository](https://github.com/newton-physics/newton-governance) for more information about project governance.
