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
# LEVEL 3 WUBU ENGINE: PHASE ENTANGLEMENT & HYPERBOLIC METRIC
# =====================================================================

@jax.jit
def hyperbolic_distance_sq(x: jnp.ndarray, y: jnp.ndarray, c: float) -> jnp.ndarray:
    """Computes true hyperbolic distance squared between two points in H^n."""
    # Using the Poincare ball distance formula
    num = jnp.sum((x - y)**2, axis=-1)
    den_x = 1.0 - c * jnp.sum(x**2, axis=-1)
    den_y = 1.0 - c * jnp.sum(y**2, axis=-1)
    
    # Clip denominators to prevent division by zero near boundary
    den_x = jnp.clip(den_x, a_min=1e-5)
    den_y = jnp.clip(den_y, a_min=1e-5)
    
    # arccosh(1 + 2 * num / (den_x * den_y))
    arg = 1.0 + 2.0 * num / (den_x * den_y)
    dist = jnp.arccosh(jnp.clip(arg, a_min=1.0001))
    return dist**2

@jax.jit
def wubu_level_3_entanglement(manifold_state: jnp.ndarray, phases: jnp.ndarray, base_coupling: float, c: float) -> jnp.ndarray:
    """
    SHELL 3: Topological Phase Entanglement.
    Uses hyperbolic distance (absorbing 2^N scaling) AND phase interference.
    """
    # 1. Hyperbolic Topological Distance Matrix
    x_exp = jnp.expand_dims(manifold_state, axis=1) # (N, 1, 3)
    y_exp = jnp.expand_dims(manifold_state, axis=0) # (1, N, 3)
    hyp_dist_sq = hyperbolic_distance_sq(x_exp, y_exp, c)
    
    # 2. Phase Interference Matrix: cos(phi_i - phi_j)
    phi_exp_1 = jnp.expand_dims(phases, axis=1)
    phi_exp_2 = jnp.expand_dims(phases, axis=0)
    interference = jnp.cos(phi_exp_1 - phi_exp_2)
    
    # 3. Combined Coupling: Decays with hyperbolic distance, modulated by phase
    W = jnp.exp(-hyp_dist_sq * 1.5) * interference * base_coupling
    
    # Remove self-coupling
    W = W * (1.0 - jnp.eye(manifold_state.shape[0]))
    return W

@jax.jit
def phase_coupled_hamiltonian(v_tangent: jnp.ndarray, phases: jnp.ndarray, W_matrix: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Flow driven by phase-modulated topological entanglement.
    """
    # Tangent flow
    flow = jnp.dot(W_matrix, v_tangent)
    
    # Phase evolution (Schrodinger-like drift based on local energy)
    local_energy = -jnp.sum(W_matrix * jnp.dot(v_tangent, v_tangent.T), axis=-1)
    phase_flow = local_energy * 0.1 # Phase shifts based on coupling environment
    
    # System Energy
    total_energy = -0.5 * jnp.sum(local_energy)
    
    return flow, phase_flow, total_energy

@jax.jit
def dynamic_step(
    v_tangent: jnp.ndarray, 
    phases: jnp.ndarray,
    key: jax.Array, 
    t: float, 
    c: float, 
    s: float, 
    base_q: jnp.ndarray
) -> dict:
    dt = 0.05
    
    # 1. Map current tangent state to Poincare Manifold for accurate distance measurement
    manifold_state = exp_map_origin(v_tangent, c, s)
    
    # 2. Level 3 Entanglement Matrix
    base_coupling = 0.6 + 0.1 * jnp.sin(t*0.5)
    W_matrix = wubu_level_3_entanglement(manifold_state, phases, base_coupling, c)
    
    # 3. Hamiltonian Flow (Vector + Phase)
    v_flow, phase_flow, energy = phase_coupled_hamiltonian(v_tangent, phases, W_matrix)
    v_evolved = v_tangent + dt * v_flow
    phases_evolved = phases + dt * phase_flow
    
    # Keep phases bounded [-pi, pi]
    phases_evolved = (phases_evolved + jnp.pi) % (2 * jnp.pi) - jnp.pi
    
    # 4. Continuous SU(2) Gate
    q_drift = jnp.array([jnp.cos(t*0.1), jnp.sin(t*0.2), jnp.sin(t*0.1), 0.0])
    current_q = base_q + q_drift
    current_q = current_q / jnp.linalg.norm(current_q)
    v_rotated = apply_su2_rotation(v_evolved, current_q)
    
    # 5. Born Rule Shimmer
    v_norm = jnp.linalg.norm(v_rotated, axis=-1, keepdims=True) + 1e-7
    P_z = 0.5 * (1.0 + (v_rotated / v_norm)[:, 2:3]) 
    uncertainty = jnp.sqrt(P_z * (1.0 - P_z) + 1e-7)
    key, subkey = jax.random.split(key)
    v_final_tangent = v_rotated + jax.random.normal(subkey, v_rotated.shape) * uncertainty * 0.12
    
    return {
        "manifold_state": exp_map_origin(v_final_tangent, c, s),
        "tangent_state": v_final_tangent,
        "phases": phases_evolved,
        "q_t": current_q,
        "energy": energy,
        "coupling": base_coupling,
        "interference_mean": jnp.mean(jnp.cos(phases_evolved)),
        "flow_magnitude": jnp.mean(jnp.linalg.norm(v_flow, axis=-1))
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
        cam_t = time.time() * 0.15
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
    
    layout["header"].update(Panel(Text(f"WuBu L3 Topological Quantum Engine | Step: {step} | t={t:.2f}s", style="bold cyan", justify="center")))
    
    table = Table(title="Phase-Coupled Hyperbolic Telemetry", expand=True)
    table.add_column("System State", justify="center", style="cyan")
    table.add_column("Interference (cos φ)", justify="right", style="magenta")
    table.add_column("Energy (H)", justify="right", style="red")
    table.add_column("L1 Flow Mag", justify="right", style="yellow")
    table.add_column("Global SU(2) Gate", justify="center", style="blue")

    q = np.array(state_dict["q_t"])
    table.add_row(
        "Phase Syncing", 
        f"{float(state_dict['interference_mean']):.4f}", 
        f"{float(state_dict['energy']):.3f}", 
        f"{float(state_dict['flow_magnitude']):.4f}", 
        f"[{q[0]:.2f}, {q[1]:.2f}, {q[2]:.2f}, {q[3]:.2f}]"
    )
        
    layout["telemetry"].update(Panel(table, title="Level 3 Thermodynamic State"))
    
    art_3d = render_3d_braille(points_3d)
    layout["visuals"].update(Panel(Align.center(Text(art_3d, style="bold bright_cyan")), title="3D Interference Domains"))
    
    footer_txt = f"LEVEL 3 ACTIVE. Entanglement mapped via Hyperbolic Distance + Phase Interference.\n"
    footer_txt += f"Exponential complexity absorbed by Poincare geometry. Hamiltonian time-slicing engaged."
    layout["footer"].update(Panel(footer_txt, style="green"))

    return layout

# =====================================================================
# EXECUTION
# =====================================================================

def main():
    console = Console()
    key = jax.random.PRNGKey(99)
    batch_size = 256 
    
    # Initialize random scattered spins AND random phases
    v_tangent = jax.random.normal(key, (batch_size, 3)) * 0.8
    key, subkey = jax.random.split(key)
    phases = jax.random.uniform(subkey, (batch_size,), minval=-jnp.pi, maxval=jnp.pi)
    
    c, s = 1.0, 1.0
    key, subkey = jax.random.split(key)
    base_q = jax.random.normal(subkey, (4,))
    base_q = base_q / jnp.linalg.norm(base_q)
    
    t, dt = 0.0, 0.05
    
    console.print("[bold yellow]JIT Compiling WuBu L3 Topological Engine...[/]")
    _ = dynamic_step(v_tangent, phases, key, t, c, s, base_q)
    
    with Live(console=console, refresh_per_second=24) as live:
        for step in range(1, 2500):
            key, subkey = jax.random.split(key)
            
            out_dict = dynamic_step(v_tangent, phases, subkey, t, c, s, base_q)
            v_tangent = out_dict["tangent_state"]
            phases = out_dict["phases"]
            t += dt
            
            points_3d = np.array(v_tangent)
            live.update(create_dashboard(step, t, out_dict, points_3d))
            time.sleep(0.03)

if __name__ == "__main__":
    main()