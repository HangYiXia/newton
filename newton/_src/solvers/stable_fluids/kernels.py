import warp as wp

# =========================================================================
# 辅助函数：3D 坐标 <-> 1D 数组偏移索引
# =========================================================================
@wp.func
def get_idx(x: int, y: int, z: int, dim: wp.vec3i, offset: int) -> int:
    """将局部的 3D 网格坐标转换为全局 1D State 数组的索引，处理越界边界"""
    # 钳制在边界内 (Clamp to edge)
    cx = wp.clamp(x, 0, dim[0] - 1)
    cy = wp.clamp(y, 0, dim[1] - 1)
    cz = wp.clamp(z, 0, dim[2] - 1)
    return offset + cx * dim[1] * dim[2] + cy * dim[2] + cz

@wp.func
def sample_field_vec3(field: wp.array(dtype=wp.vec3), p: wp.vec3, dim: wp.vec3i, offset: int) -> wp.vec3:
    """对三维向量场进行简单的三线性插值 (Trilinear Interpolation)"""
    x = p[0]
    y = p[1]
    z = p[2]
    
    x0 = int(wp.floor(x))
    y0 = int(wp.floor(y))
    z0 = int(wp.floor(z))
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    tx = x - float(x0)
    ty = y - float(y0)
    tz = z - float(z0)

    # 采样 8 个顶点 (为保持代码简洁，这里演示最基础的读取，实际可展开写完整的三线性插值)
    c000 = field[get_idx(x0, y0, z0, dim, offset)]
    c100 = field[get_idx(x1, y0, z0, dim, offset)]
    c010 = field[get_idx(x0, y1, z0, dim, offset)]
    c110 = field[get_idx(x1, y1, z0, dim, offset)]
    c001 = field[get_idx(x0, y0, z1, dim, offset)]
    c101 = field[get_idx(x1, y0, z1, dim, offset)]
    c011 = field[get_idx(x0, y1, z1, dim, offset)]
    c111 = field[get_idx(x1, y1, z1, dim, offset)]

    # 三线性插值混合...
    c00 = c000 * (1.0 - tx) + c100 * tx
    c10 = c010 * (1.0 - tx) + c110 * tx
    c01 = c001 * (1.0 - tx) + c101 * tx
    c11 = c011 * (1.0 - tx) + c111 * tx

    c0 = c00 * (1.0 - ty) + c10 * ty
    c1 = c01 * (1.0 - ty) + c11 * ty

    return c0 * (1.0 - tz) + c1 * tz

# =========================================================================
# 1. 平流步 (Advection) - 使用半拉格朗日回溯
# =========================================================================
@wp.kernel
def advect_vel_kernel(
    vel_prev: wp.array(dtype=wp.vec3),
    vel_out: wp.array(dtype=wp.vec3),
    dim: wp.vec3i,
    dx: float,
    dt: float,
    offset: int
):
    i, j, k = wp.tid()
    idx = get_idx(i, j, k, dim, offset)
    
    # 当前速度
    v = vel_prev[idx]
    
    # 逆向追踪 (Backtrace)，计算回退到的网格浮点坐标
    # 注意：如果网格带 Transform，这里应转换到局部空间，简单起见假设格子间距一致
    pos = wp.vec3(float(i), float(j), float(k))
    pos_prev = pos - v * (dt / dx)
    
    # 插值旧速度场
    vel_out[idx] = sample_field_vec3(vel_prev, pos_prev, dim, offset)

# =========================================================================
# 2. 外力步 (External Forces)
# =========================================================================
@wp.kernel
def apply_forces_kernel(
    vel: wp.array(dtype=wp.vec3),
    gravity: wp.vec3,
    dt: float,
    offset: int,
    dim: wp.vec3i
):
    i, j, k = wp.tid()
    idx = get_idx(i, j, k, dim, offset)
    # 简单叠加重力
    vel[idx] = vel[idx] + gravity * dt

# =========================================================================
# 3. 散度计算 (Divergence)
# =========================================================================
@wp.kernel
def compute_divergence_kernel(
    vel: wp.array(dtype=wp.vec3),
    div: wp.array(dtype=wp.float32),
    dim: wp.vec3i,
    dx: float,
    offset: int
):
    i, j, k = wp.tid()
    idx = get_idx(i, j, k, dim, offset)

    # 中心差分计算散度
    v_L = vel[get_idx(i - 1, j, k, dim, offset)][0]
    v_R = vel[get_idx(i + 1, j, k, dim, offset)][0]
    v_D = vel[get_idx(i, j - 1, k, dim, offset)][1]
    v_U = vel[get_idx(i, j + 1, k, dim, offset)][1]
    v_B = vel[get_idx(i, j, k - 1, dim, offset)][2]
    v_F = vel[get_idx(i, j, k + 1, dim, offset)][2]

    # div = 0.5 / dx * ((R - L) + (U - D) + (F - B))
    div[idx] = 0.5 * ((v_R - v_L) + (v_U - v_D) + (v_F - v_B)) / dx

# =========================================================================
# 4. 泊松求解 (Jacobi Iteration for Pressure)
# =========================================================================
@wp.kernel
def jacobi_pressure_kernel(
    p_in: wp.array(dtype=wp.float32),
    p_out: wp.array(dtype=wp.float32),
    div: wp.array(dtype=wp.float32),
    dim: wp.vec3i,
    dx: float,
    offset: int
):
    i, j, k = wp.tid()
    idx = get_idx(i, j, k, dim, offset)

    p_L = p_in[get_idx(i - 1, j, k, dim, offset)]
    p_R = p_in[get_idx(i + 1, j, k, dim, offset)]
    p_D = p_in[get_idx(i, j - 1, k, dim, offset)]
    p_U = p_in[get_idx(i, j + 1, k, dim, offset)]
    p_B = p_in[get_idx(i, j, k - 1, dim, offset)]
    p_F = p_in[get_idx(i, j, k + 1, dim, offset)]

    # Jacobi 迭代公式
    d = div[idx]
    p_out[idx] = (p_L + p_R + p_D + p_U + p_B + p_F - d * (dx * dx)) / 6.0

# =========================================================================
# 5. 速度修正 (Subtract Gradient)
# =========================================================================
@wp.kernel
def subtract_gradient_kernel(
    vel: wp.array(dtype=wp.vec3),
    p: wp.array(dtype=wp.float32),
    dim: wp.vec3i,
    dx: float,
    offset: int
):
    i, j, k = wp.tid()
    idx = get_idx(i, j, k, dim, offset)

    p_L = p[get_idx(i - 1, j, k, dim, offset)]
    p_R = p[get_idx(i + 1, j, k, dim, offset)]
    p_D = p[get_idx(i, j - 1, k, dim, offset)]
    p_U = p[get_idx(i, j + 1, k, dim, offset)]
    p_B = p[get_idx(i, j, k - 1, dim, offset)]
    p_F = p[get_idx(i, j, k + 1, dim, offset)]

    grad_x = (p_R - p_L) * 0.5 / dx
    grad_y = (p_U - p_D) * 0.5 / dx
    grad_z = (p_F - p_B) * 0.5 / dx

    v = vel[idx]
    vel[idx] = wp.vec3(v[0] - grad_x, v[1] - grad_y, v[2] - grad_z)



# 可视化粒子平流
@wp.kernel
def advect_marker_particles_kernel(
    particle_q_in: wp.array(dtype=wp.vec3),
    particle_q_out: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    grid_vel: wp.array(dtype=wp.vec3),
    grid_dim: wp.vec3i,
    grid_dx: float,
    grid_transform: wp.transform,
    grid_offset: int,
    dt: float,
    particle_offset: int  # 对应不同 World 的粒子起始索引
):
    # 线程 ID 对应当前 World 的局部粒子索引
    tid = wp.tid()
    p_idx = particle_offset + tid
    
    # 略过被禁用的粒子
    if particle_flags[p_idx] == 0:
        return

    # 1. 获取世界坐标，转换到流体网格的局部坐标系
    p_world = particle_q_in[p_idx]
    inv_tf = wp.transform_inverse(grid_transform)
    p_local = wp.transform_point(inv_tf, p_world)
    
    # 局部连续浮点网格坐标 (用于插值)
    p_grid = p_local / grid_dx

    # ==========================================
    # RK2 (Runge-Kutta 2阶) 积分防止粒子飞出涡流
    # ==========================================
    # k1 (当前点速度)
    v_local_1 = sample_field_vec3(grid_vel, p_grid, grid_dim, grid_offset)
    
    # 预测半步位置
    p_local_mid = p_local + v_local_1 * (0.5 * dt)
    p_grid_mid = p_local_mid / grid_dx
    
    # k2 (中点速度)
    v_local_2 = sample_field_vec3(grid_vel, p_grid_mid, grid_dim, grid_offset)
    
    # 实际推进
    p_local_new = p_local + v_local_2 * dt
    
    # 将更新后的局部坐标转回世界坐标
    p_world_new = wp.transform_point(grid_transform, p_local_new)
    
    particle_q_out[p_idx] = p_world_new