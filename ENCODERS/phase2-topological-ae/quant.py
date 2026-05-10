import os
import time
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.align import Align
from rich.text import Text

os.environ['JAX_PLATFORMS'] = 'cpu'

# =====================================================================
# RIGOROUS HYPERBOLIC & QUANTUM MATH CORE
# =====================================================================

def exp_map_origin(v: jnp.ndarray, c: jnp.ndarray, s: jnp.ndarray) -> jnp.ndarray:
    v_norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
    v_norm = jnp.clip(v_norm, a_min=1e-7)
    sqrt_c = jnp.sqrt(c)
    return jnp.tanh(s * (sqrt_c * v_norm / 2.0)) * (v / (sqrt_c * v_norm))

def log_map_origin(x: jnp.ndarray, c: jnp.ndarray, s: jnp.ndarray) -> jnp.ndarray:
    x_norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
    x_norm = jnp.clip(x_norm, a_min=1e-7, a_max=1.0 - 1e-7)
    sqrt_c = jnp.sqrt(c)
    return (2.0 / (s * sqrt_c)) * jnp.arctanh(sqrt_c * x_norm) * (x / x_norm)

def apply_su2_rotation(v_tangent: jnp.ndarray, q: jnp.ndarray) -> jnp.ndarray:
    q_norm = q / jnp.linalg.norm(q, keepdims=True)
    w, x, y, z = q_norm[0], q_norm[1], q_norm[2], q_norm[3]
    R = jnp.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),       1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),       1 - 2*(x**2 + y**2)]
    ])
    return jnp.dot(v_tangent, R.T)

def compute_varentropy(state: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
    probs = jax.nn.softmax(state / jnp.clip(sigma, 1e-5, 10.0))
    log_probs = jnp.log(probs + 1e-9)
    entropy = -jnp.sum(probs * log_probs, axis=-1)
    sq_entropy = jnp.sum(probs * (log_probs ** 2), axis=-1)
    return sq_entropy - (entropy ** 2)

# =====================================================================
# DYNAMICAL WUBU ENGINE
# =====================================================================

@jax.jit
def hamiltonian_flow(v: jnp.ndarray) -> jnp.ndarray:
    """Non-linear continuous flow vector field in T_o(H^3)."""
    # Simulate an attractor basin + orbital momentum
    r_sq = jnp.sum(v**2, axis=-1, keepdims=True)
    attractor = -0.5 * v * (r_sq - 0.5) # Pulls towards radius 0.707
    
    # Cross product with Z axis for orbital rotation
    z_axis = jnp.array([0.0, 0.0, 1.0])
    orbital = jnp.cross(v, z_axis) * 2.0
    
    return attractor + orbital

@jax.jit
def dynamic_shimmer_step(
    v_tangent: jnp.ndarray, 
    key: jax.Array, 
    t: float, 
    c: jnp.ndarray, 
    s: jnp.ndarray, 
    base_sigma: jnp.ndarray,
    base_q: jnp.ndarray
) -> dict:
    # 1. Parameter Evolution (Time-dependent drift)
    # Sigma breathes over time
    current_sigma = base_sigma * (1.0 + 0.3 * jnp.sin(t * 0.5))
    
    # Q-Gate rotates continuously
    q_drift = jnp.array([jnp.cos(t*0.2), jnp.sin(t*0.2), jnp.sin(t*0.1), 0.0])
    current_q = base_q + q_drift
    current_q = current_q / jnp.linalg.norm(current_q)

    # 2. Tangent Space Integration (dt = 0.05)
    dt = 0.05
    flow = hamiltonian_flow(v_tangent)
    v_evolved = v_tangent + dt * flow
    
    # 3. Apply Continuous SU(2) Gate
    v_rotated = apply_su2_rotation(v_evolved, current_q)
    
    # 4. Inject Quantum Variance (Shimmer)
    quantum_noise = jax.random.normal(key, v_rotated.shape) * current_sigma * 0.1
    v_final_tangent = v_rotated + quantum_noise
    
    # 5. Map back to manifold
    manifold_state = exp_map_origin(v_final_tangent, c, s)
    varentropy = compute_varentropy(v_final_tangent, current_sigma)

    return {
        "manifold_state": manifold_state,
        "tangent_state": v_final_tangent,
        "sigma_t": current_sigma,
        "q_t": current_q,
        "varentropy": jnp.mean(varentropy),
        "flow_magnitude": jnp.mean(jnp.linalg.norm(flow, axis=-1))
    }

# =====================================================================
# HIGH-RES 3D BRAILLE RENDERER
# =====================================================================

def render_3d_braille(points_3d: np.ndarray, width=60, height=30) -> str:
    """
    Projects 3D points via perspective matrix and renders to sub-pixel Braille.
    Braille characters are 2x4 dot matrices. 
    Internal canvas is (width*2) x (height*4).
    """
    canvas_w = width * 2
    canvas_h = height * 4
    canvas = np.zeros((canvas_h, canvas_w), dtype=bool)
    
    # Simple perspective projection
    fov = 2.0
    z_offset = 3.0
    
    for pt in points_3d:
        x, y, z = pt[0], pt[1], pt[2]
        
        # Apply slight isometric tilt
        tilt_x = x * np.cos(0.5) - z * np.sin(0.5)
        tilt_z = x * np.sin(0.5) + z * np.cos(0.5) + z_offset
        tilt_y = y
        
        if tilt_z < 0.1: continue # Behind camera
        
        # Project to 2D
        proj_x = (tilt_x / tilt_z) * fov
        proj_y = (tilt_y / tilt_z) * fov
        
        # Map to canvas coordinates
        px = int((proj_x + 1.0) * 0.5 * canvas_w)
        py = int((proj_y + 1.0) * 0.5 * canvas_h)
        
        if 0 <= px < canvas_w and 0 <= py < canvas_h:
            canvas[py, px] = True

    # Convert binary matrix to Braille Unicode
    # Braille offset is 0x2800. Dots are mapped by specific bit flags.
    braille_map = np.array([[1, 8], [2, 16], [4, 32], [64, 128]])
    lines = []
    
    for y in range(0, canvas_h, 4):
        line = []
        for x in range(0, canvas_w, 2):
            block = canvas[y:y+4, x:x+2]
            # Pad if block is cut off at edges
            if block.shape != (4, 2):
                padded = np.zeros((4,2), dtype=bool)
                padded[:block.shape[0], :block.shape[1]] = block
                block = padded
            char_val = 0x2800 + np.sum(block * braille_map)
            line.append(chr(char_val))
        lines.append("".join(line))
        
    return "\n".join(lines)

# =====================================================================
# TERMINAL UI
# =====================================================================

def create_dashboard(step: int, t: float, state_dict: dict, points_3d: np.ndarray) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=5)
    )
    layout["main"].split_row(
        Layout(name="telemetry", ratio=2),
        Layout(name="visuals", ratio=2)
    )
    
    layout["header"].update(Panel(Text(f"WuBu Quantum Engine | Step: {step} | t={t:.2f}s | O(n) Continuous Integration", style="bold cyan", justify="center")))
    
    table = Table(title="Live Manifold Telemetry", expand=True)
    table.add_column("Vector Flow", justify="center", style="cyan")
    table.add_column("Breathing (σ_t)", justify="right", style="magenta")
    table.add_column("Varentropy", justify="right", style="yellow")
    table.add_column("Hamiltonian Mag", justify="right", style="red")
    table.add_column("SU(2) Drift Gate", justify="center", style="blue")

    q = np.array(state_dict["q_t"])
    table.add_row(
        "Active Integration", 
        f"{float(state_dict['sigma_t'][0]):.5f}", 
        f"{float(state_dict['varentropy']):.5f}", 
        f"{float(state_dict['flow_magnitude']):.5f}", 
        f"[{q[0]:.2f}, {q[1]:.2f}, {q[2]:.2f}, {q[3]:.2f}]"
    )
        
    layout["telemetry"].update(Panel(table, title="Dynamical State Data"))
    
    # 3D Render
    art_3d = render_3d_braille(points_3d, width=45, height=20)
    layout["visuals"].update(Panel(Align.center(Text(art_3d, style="bold bright_green")), title="3D Perspective Slice (Sub-Pixel Tensor Projection)"))
    
    footer_txt = f"HAMILTONIAN INTEGRATOR RUNNING. Translational velocity mapped to 3D Braille Tensor.\n"
    footer_txt += f"Raw Coordinates (Node 0): [X: {points_3d[0,0]:.3f}, Y: {points_3d[0,1]:.3f}, Z: {points_3d[0,2]:.3f}]"
    layout["footer"].update(Panel(footer_txt, style="green"))

    return layout

# =====================================================================
# EXECUTION
# =====================================================================

def main():
    console = Console()
    key = jax.random.PRNGKey(42)
    batch_size = 256 # More nodes to show off 3D geometry
    
    # Initialize a cluster of points in tangent space
    v_tangent = jax.random.normal(key, (batch_size, 3)) * 0.5
    
    # Base parameters
    c = jnp.array([1.0])
    s = jnp.array([1.0])
    base_sigma = jnp.array([0.5])
    base_q = jax.random.normal(key, (4,))
    base_q = base_q / jnp.linalg.norm(base_q)
    
    t = 0.0
    dt = 0.05
    
    with Live(console=console, refresh_per_second=20) as live:
        for step in range(1, 1000):
            key, subkey = jax.random.split(key)
            
            # Step the dynamical system
            out_dict = dynamic_shimmer_step(
                v_tangent, subkey, t, c, s, base_sigma, base_q
            )
            
            # Update state for next tick
            v_tangent = out_dict["tangent_state"]
            t += dt
            
            # Extract 3D points for rendering
            points_3d = np.array(v_tangent)
            
            # UI Update
            live.update(create_dashboard(step, t, out_dict, points_3d))
            time.sleep(0.04)

if __name__ == "__main__":
    main()