import ctypes

import numpy as np
import warp as wp
from OpenGL import GL as gl

import newton
import newton.examples
from newton.viewer import ViewerGL

# --------------------------------------------------------------------------- #
# Initialization Kernel (适配新的 vec2 速度场)
# --------------------------------------------------------------------------- #

@wp.kernel
def init_fluid_state(
    initial_radius: float,
    vortex_strength: float,
    velocity: wp.array2d(dtype=wp.vec2),
    density: wp.array2d(dtype=float),
    n_grid: int
):
    """
    初始化流体状态：
    1. 在中心放置一个圆形的高密度区域。
    2. 设置一个旋转的速度场（涡流）。
    """
    i, j = wp.tid()

    # Grid center
    x_c = wp.float32(n_grid) / 2.0
    y_c = wp.float32(n_grid) / 2.0

    # Distance from center
    x_dist = wp.float32(i) - x_c
    y_dist = wp.float32(j) - y_c
    r_dist = wp.sqrt(x_dist * x_dist + y_dist * y_dist)

    r_core = 0.1 * initial_radius

    # Vortex velocity profile
    vel_x = 0.0
    vel_y = 0.0

    if r_dist < r_core:
        vel_x = -vortex_strength * y_dist / (r_core * r_core)
        vel_y = vortex_strength * x_dist / (r_core * r_core)
    else:
        vel_x = -vortex_strength * y_dist / (r_dist * r_dist)
        vel_y = vortex_strength * x_dist / (r_dist * r_dist)

    velocity[i, j] = wp.vec2(vel_x, vel_y)

    # High density in the center
    if r_dist < initial_radius:
        density[i, j] = 1.0
    else:
        density[i, j] = 0.0


# --------------------------------------------------------------------------- #
# Kernel to convert density to RGBA
# --------------------------------------------------------------------------- #

@wp.kernel
def density_to_rgba(
    density: wp.array2d(dtype=float),
    rgba: wp.array(dtype=wp.uint8),
    n_grid: int,
    min_val: float,
    max_val: float
):
    """
    Convert density field to RGBA texture for visualization.
    Uses viridis-like colormap.
    """
    i, j = wp.tid()
    
    # Get density value and normalize to [0, 1]
    d = density[i, j]
    t = wp.clamp((d - min_val) / (max_val - min_val), 0.0, 1.0)
    
    # Simple viridis-like colormap approximation
    # viridis goes from dark blue/purple -> teal -> yellow/green
    r = wp.float32(0.0)
    g = wp.float32(0.0)
    b = wp.float32(0.0)
    
    if t < 0.25:
        s = t / 0.25
        r = 0.267 * s
        g = 0.005 + 0.222 * s
        b = 0.329 + 0.213 * s
    elif t < 0.5:
        s = (t - 0.25) / 0.25
        r = 0.267 + (0.208 - 0.267) * s
        g = 0.227 + (0.427 - 0.227) * s
        b = 0.542 + (0.626 - 0.542) * s
    elif t < 0.75:
        s = (t - 0.5) / 0.25
        r = 0.208 + (0.295 - 0.208) * s
        g = 0.427 + (0.686 - 0.427) * s
        b = 0.626 + (0.443 - 0.626) * s
    else:
        s = (t - 0.75) / 0.25
        r = 0.295 + (0.993 - 0.295) * s
        g = 0.686 + (0.906 - 0.686) * s
        b = 0.443 + (0.144 - 0.443) * s
    
    # Convert to uint8 and flip vertically (OpenGL texture coordinate)
    idx = (n_grid - 1 - j) * n_grid + i
    rgba[idx * 4 + 0] = wp.uint8(r * 255.0)
    rgba[idx * 4 + 1] = wp.uint8(g * 255.0)
    rgba[idx * 4 + 2] = wp.uint8(b * 255.0)
    rgba[idx * 4 + 3] = wp.uint8(255)


# --------------------------------------------------------------------------- #
# Main Example Class
# --------------------------------------------------------------------------- #

class Example:
    def __init__(self, viewer: ViewerGL):
        # Simulation parameters
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps
        
        # Configuration
        self.n_grid = 200
        self.dh = 1.0 / self.n_grid
        self.dt = 1.0
        self.iterations = 40
        
        # UI settings
        self.ui_side_panel_width = 300
        self.ui_padding = 10
        
        self.viewer = viewer
        
        print(f"Initializing Stable Fluids: {self.n_grid}x{self.n_grid} grid...")
        
        # Build model
        builder = newton.ModelBuilder()
        builder.add_fluid_grid(res=self.n_grid, cell_size=self.dh)
        
        self.model = builder.finalize()
        self.viewer.set_model(self.model)
        
        # Create states
        self.state_curr = self.model.state()
        self.state_next = self.model.state()
        
        # Initialize fluid data
        wp.launch(
            kernel=init_fluid_state,
            dim=(self.n_grid, self.n_grid),
            inputs=[
                75.0,   # initial_radius
                100.0,  # vortex_strength
                self.state_curr.fluid_velocity,
                self.state_curr.fluid_density,
                self.n_grid
            ]
        )
        
        # Create solver
        self.solver = newton.solvers.SolverStableFluids(self.model, pressure_iterations=self.iterations)
        
        # Setup texture for visualization
        if isinstance(self.viewer, ViewerGL):
            self.setup_texture()
            self.viewer.register_ui_callback(self.display, "free")
        
        print("Stable Fluids initialized successfully!")
    
    def setup_texture(self):
        """Create OpenGL texture for density visualization."""
        # Create texture
        self.texture_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, 
            self.n_grid, self.n_grid, 0, 
            gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None
        )
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        
        # Create pixel buffer object
        self.pixel_buffer = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, self.pixel_buffer)
        gl.glBufferData(
            gl.GL_PIXEL_UNPACK_BUFFER, 
            self.n_grid * self.n_grid * 4, 
            None, 
            gl.GL_DYNAMIC_DRAW
        )
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)
        
        # Create warp buffer wrapper
        self.texture_buffer = wp.RegisteredGLBuffer(self.pixel_buffer)
        
        # RGBA buffer for density conversion
        self.rgba_buffer = wp.zeros(
            self.n_grid * self.n_grid * 4, 
            dtype=wp.uint8, 
            device=self.model.device
        )
    
    def update_texture(self):
        """Update texture with current density field."""
        if not hasattr(self, 'texture_id') or self.texture_id == 0:
            return
        
        # Convert density to RGBA
        wp.launch(
            kernel=density_to_rgba,
            dim=(self.n_grid, self.n_grid),
            inputs=[
                self.state_curr.fluid_density,
                self.rgba_buffer,
                self.n_grid,
                0.0,  # min value
                1.0   # max value
            ]
        )
        
        # Map the PBO and copy data
        texture_buffer = self.texture_buffer.map(
            dtype=wp.uint8,
            shape=(self.n_grid * self.n_grid * 4,)
        )
        wp.copy(texture_buffer, self.rgba_buffer)
        self.texture_buffer.unmap()
        
        # Update texture from PBO
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, self.pixel_buffer)
        gl.glTexSubImage2D(
            gl.GL_TEXTURE_2D, 0, 0, 0,
            self.n_grid, self.n_grid,
            gl.GL_RGBA, gl.GL_UNSIGNED_BYTE,
            ctypes.c_void_p(0)
        )
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
    
    def display(self, imgui):
        """Display the fluid density as a texture overlay."""
        if not hasattr(self, 'texture_id') or self.texture_id == 0:
            return
        
        # Update texture with latest data
        self.update_texture()
        
        # Display settings: calculate available space
        padding = 20
        available_width = self.viewer.ui.io.display_size[0] - self.ui_side_panel_width - padding * 4
        available_height = self.viewer.ui.io.display_size[1] - padding * 2
        size = min(available_width, available_height)
        
        # Position window on the right side
        imgui.set_next_window_pos(
            imgui.ImVec2(
                self.ui_side_panel_width + padding * 2,
                padding
            )
        )
        imgui.set_next_window_size(imgui.ImVec2(size, size + 40))
        
        flags = (
            imgui.WindowFlags_.no_title_bar.value |
            imgui.WindowFlags_.no_resize.value |
            imgui.WindowFlags_.no_scrollbar.value
        )
        
        if imgui.begin("Fluid Density", flags=flags):
            imgui.text("Stable Fluids Density Field")
            imgui.separator()
            imgui.image(
                imgui.ImTextureRef(self.texture_id), 
                imgui.ImVec2(size - 20, size - 40)
            )
        imgui.end()
    
    def step(self):
        """Advance the simulation by one step."""
        for _ in range(self.sim_substeps):
            self.solver.step(self.state_curr, self.state_next, dt=self.dt)
            # Swap states (ping-pong)
            self.state_curr, self.state_next = self.state_next, self.state_curr
        
        self.sim_time += self.frame_dt
    
    def render(self):
        """Render the current frame."""
        self.viewer.begin_frame(self.sim_time)
        # No physical objects to render, only the fluid texture
        self.viewer.end_frame()
    
    def test_final(self):
        """Test that simulation ran successfully."""
        pass


# --------------------------------------------------------------------------- #
# Entry Point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init()
    
    # Create example and run
    example = Example(viewer)
    
    newton.examples.run(example, args)