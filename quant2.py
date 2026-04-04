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
from typing import Tuple
os.environ['JAX_PLATFORMS'] = 'cpu'

# =====================================================================
# RIGOROUS HYPERBOLIC MATH CORE
# =====================================================================

def exp_map_origin(v: jnp.ndarray, c: float, s: float) -> jnp.ndarray:
    v_norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
    v_norm = jnp.clip(v_norm, a_min=1e-7)
    sqrt_c = jnp.sqrt(c)
    return jnp.tanh(s * (sqrt_c * v_norm / 2.0)) * (v / (sqrt_c * v_norm))

def log_map_origin(x: jnp.ndarray, c: float, s: float) -> jnp.ndarray:
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

# =====================================================================
# NESTED WUBU QUANTUM ENGINE (HEISENBERG / ISING GRAPH)
# =====================================================================

@jax.jit
def wubu_level_2_entanglement(v_tangent: jnp.ndarray, base_coupling: float) -> jnp.ndarray:
    """
    SHELL 2: Computes Relative Vectors (d_ij) to define entanglement links.
    Creates a dynamic adjacency matrix where close nodes in tangent space couple strongly.
    """
    # Compute pairwise relative distance matrix ||v_i - v_j||^2
    v_expanded_1 = jnp.expand_dims(v_tangent, axis=1) # (N, 1, 3)
    v_expanded_2 = jnp.expand_dims(v_tangent, axis=0) # (1, N, 3)
    dist_sq = jnp.sum((v_expanded_1 - v_expanded_2)**2, axis=-1)
    
    # Entanglement strength decays exponentially with topological distance
    W = jnp.exp(-dist_sq * 2.0) * base_coupling
    
    # Remove self-coupling (diagonal)
    W = W * (1.0 - jnp.eye(v_tangent.shape[0]))
    return W

@jax.jit
def hamiltonian_flow_heisenberg(v_tangent: jnp.ndarray, W_matrix: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    SHELL 1: Continuous Heisenberg Model Flow.
    Spins align based on the Shell 2 Entanglement Matrix.
    """
    # Flow = J * Sum(W_ij * v_j) -> nodes pull each other into aligned domains
    flow = jnp.dot(W_matrix, v_tangent)
    
    # Optional boundary constraint (attractor to prevent infinity blowout)
    r_sq = jnp.sum(v_tangent**2, axis=-1, keepdims=True)
    attractor = -0.1 * v_tangent * (r_sq) 
    
    total_flow = flow + attractor
    
    # Compute system energy: H = -Sum(W_ij * (v_i dot v_j))
    energy = -0.5 * jnp.sum(W_matrix * jnp.dot(v_tangent, v_tangent.T))
    return total_flow, energy

@jax.jit
def born_rule_shimmer(v_tangent: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
    """
    BORN RULE: Variance based on measurement probability, not static Gaussian.
    """
    # Normalize state to calculate probabilities
    v_norm = jnp.linalg.norm(v_tangent, axis=-1, keepdims=True) + 1e-7
    v_unit = v_tangent / v_norm
    
    # Probability of measuring |1> along Z-axis (simplified Bloch projection)
    P_z = 0.5 * (1.0 + v_unit[:, 2:3]) 
    
    # Quantum uncertainty: max at P=0.5 (superposition), zero at P=0 or 1 (certainty)
    uncertainty = jnp.sqrt(P_z * (1.0 - P_z) + 1e-7)
    
    # Noise injected strictly proportional to uncertainty
    noise = jax.random.normal(key, v_tangent.shape) * uncertainty * 0.15
    return v_tangent + noise

@jax.jit
def dynamic_step(
    v_tangent: jnp.ndarray, 
    key: jax.Array, 
    t: float, 
    c: float, 
    s: float, 
    base_q: jnp.ndarray
) -> dict:
    dt = 0.04
    
    # 1. Level 2 (Entanglement Mapping via Relative Vectors)
    base_coupling = 0.5 + 0.2 * jnp.sin(t) # Pulsing coupling strength
    W_matrix = wubu_level_2_entanglement(v_tangent, base_coupling)
    
    # 2. Level 1 (Hamiltonian Flow)
    flow, energy = hamiltonian_flow_heisenberg(v_tangent, W_matrix)
    v_evolved = v_tangent + dt * flow
    
    # 3. Continuous SU(2) Gate (Global rotation pulse)
    q_drift = jnp.array([jnp.cos(t*0.1), jnp.sin(t*0.3), jnp.sin(t*0.2), 0.0])
    current_q = base_q + q_drift
    current_q = current_q / jnp.linalg.norm(current_q)
    v_rotated = apply_su2_rotation(v_evolved, current_q)
    
    # 4. Born Rule Shimmer
    key, subkey = jax.random.split(key)
    v_final_tangent = born_rule_shimmer(v_rotated, subkey)
    
    # 5. Return to Manifold H^3
    manifold_state = exp_map_origin(v_final_tangent, c, s)

    return {
        "manifold_state": manifold_state,
        "tangent_state": v_final_tangent,
        "q_t": current_q,
        "energy": energy,
        "coupling": base_coupling,
        "flow_magnitude": jnp.mean(jnp.linalg.norm(flow, axis=-1))
    }

# =====================================================================
# BRAILLE TENSOR RENDERER
# =====================================================================

def render_3d_braille(points_3d: np.ndarray, width=65, height=25) -> str:
    canvas_w, canvas_h = width * 2, height * 4
    canvas = np.zeros((canvas_h, canvas_w), dtype=bool)
    
    fov, z_offset = 1.8, 2.5
    
    for pt in points_3d:
        x, y, z = pt[0], pt[1], pt[2]
        
        # Rotating camera
        cam_t = time.time() * 0.2
        rot_x = x * np.cos(cam_t) - z * np.sin(cam_t)
        rot_z = x * np.sin(cam_t) + z * np.cos(cam_t) + z_offset
        
        if rot_z < 0.1: continue 
        
        px = int(((rot_x / rot_z) * fov + 1.0) * 0.5 * canvas_w)
        py = int(((y / rot_z) * fov + 1.0) * 0.5 * canvas_h)
        
        if 0 <= px < canvas_w and 0 <= py < canvas_h:
            canvas[py, px] = True

    braille_map = np.array([[1, 8], [2, 16], [4, 32], [64, 128]])
    lines = []
    
    for y in range(0, canvas_h, 4):
        line = []
        for x in range(0, canvas_w, 2):
            block = canvas[y:y+4, x:x+2]
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
    
    layout["header"].update(Panel(Text(f"WuBu Nested Heisenberg Engine | Step: {step} | t={t:.2f}s", style="bold cyan", justify="center")))
    
    table = Table(title="L1/L2 Nested Telemetry", expand=True)
    table.add_column("System State", justify="center", style="cyan")
    table.add_column("Ising Energy (H)", justify="right", style="red")
    table.add_column("L2 Coupling Strength", justify="right", style="magenta")
    table.add_column("L1 Flow Mag", justify="right", style="yellow")
    table.add_column("Global SU(2) Gate", justify="center", style="blue")

    q = np.array(state_dict["q_t"])
    table.add_row(
        "Active Integration", 
        f"{float(state_dict['energy']):.3f}", 
        f"{float(state_dict['coupling']):.3f}", 
        f"{float(state_dict['flow_magnitude']):.4f}", 
        f"[{q[0]:.2f}, {q[1]:.2f}, {q[2]:.2f}, {q[3]:.2f}]"
    )
        
    layout["telemetry"].update(Panel(table, title="Quantum Thermodynamic State"))
    
    art_3d = render_3d_braille(points_3d)
    layout["visuals"].update(Panel(Align.center(Text(art_3d, style="bold bright_cyan")), title="3D Spin Domains (Rotating Camera)"))
    
    footer_txt = f"LEVEL 2 NESTING ACTIVE. Entanglement mapped via relative vectors. Shimmer constrained via Born Rule probability.\n"
    footer_txt += f"Tracking 256 coupled spins. System Energy approaching local minima."
    layout["footer"].update(Panel(footer_txt, style="green"))

    return layout

# =====================================================================
# EXECUTION
# =====================================================================

def main():
    console = Console()
    key = jax.random.PRNGKey(99)
    batch_size = 256 
    
    # Initialize random scattered spins
    v_tangent = jax.random.normal(key, (batch_size, 3)) * 0.8
    
    c, s = 1.0, 1.0
    base_q = jax.random.normal(key, (4,))
    base_q = base_q / jnp.linalg.norm(base_q)
    
    t, dt = 0.0, 0.04
    
    # Pre-compile JIT
    console.print("[bold yellow]JIT Compiling WuBu L2 Engine...[/]")
    _ = dynamic_step(v_tangent, key, t, c, s, base_q)
    
    with Live(console=console, refresh_per_second=24) as live:
        for step in range(1, 2000):
            key, subkey = jax.random.split(key)
            
            out_dict = dynamic_step(v_tangent, subkey, t, c, s, base_q)
            v_tangent = out_dict["tangent_state"]
            t += dt
            
            points_3d = np.array(v_tangent)
            live.update(create_dashboard(step, t, out_dict, points_3d))
            time.sleep(0.03)

if __name__ == "__main__":
    main()