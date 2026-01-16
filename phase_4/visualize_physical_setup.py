import matplotlib.pyplot as plt
import matplotlib.patches as patches
import soundfile as sf
import numpy as np
import config
import os

def calculate_physics(ir_path, fs):
    """
    Analyzes an Impulse Response to find the delay and convert it to meters.
    Assumes Speed of Sound c = 343 m/s.
    """
    if not os.path.exists(ir_path):
        return 0, 0, 0, np.zeros(64)
        
    ir_data, _ = sf.read(ir_path)
    
    # Find the index of the loudest peak (the direct sound)
    peak_sample = np.argmax(np.abs(ir_data))
    
    # Calculate time in seconds
    delay_sec = peak_sample / fs
    
    # Calculate physical distance (d = v * t)
    distance_m = delay_sec * 343.0
    
    return peak_sample, delay_sec, distance_m, ir_data

def main():
    print("--- VISUALIZING SIMULATION SETUP ---")
    
    # 1. Calculate Physics from the WAV files
    p_sample, p_time, p_dist, p_ir = calculate_physics(config.PRIMARY_PATH_FILE, config.SAMPLE_RATE)
    s_sample, s_time, s_dist, s_ir = calculate_physics(config.SECONDARY_PATH_FILE, config.SAMPLE_RATE)
    
    # 2. Setup Plot
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"ANC Simulation Setup: Physical Layout & Acoustic Paths\nSample Rate: {config.SAMPLE_RATE}Hz | Speed of Sound: 343 m/s", fontsize=16, fontweight='bold')
    
    # --- SUBPLOT 1: PHYSICAL SCHEMATIC (Top Half) ---
    ax_map = fig.add_subplot(2, 1, 1)
    ax_map.set_title("Physical Layout Representation (Top-Down View)", fontsize=14)
    ax_map.set_xlim(-0.1, max(p_dist, s_dist) + 0.4)
    ax_map.set_ylim(-0.5, 0.5)
    ax_map.axis('off')
    
    # Draw "Car Cabin" Box
    cabin_rect = patches.Rectangle((0, -0.4), p_dist + 0.2, 0.8, linewidth=2, edgecolor='gray', facecolor='#f0f0f0', alpha=0.5, linestyle='--')
    ax_map.add_patch(cabin_rect)
    ax_map.text(0.1, -0.35, "Simulated Car Cabin", color='gray', fontsize=12)

    # Coordinates
    src_x = 0
    ear_x = p_dist
    spk_x = ear_x - s_dist # Speaker is closer to ear than engine

    # Draw Components
    # 1. Noise Source (Engine)
    ax_map.scatter(src_x, 0, s=300, c='red', edgecolors='black', zorder=10, label='Source')
    ax_map.text(src_x, 0.1, f"NOISE SOURCE\n(Engine)\nReference Mic", ha='center', fontweight='bold')

    # 2. Driver's Ear (Error Mic)
    ax_map.scatter(ear_x, 0, s=300, c='blue', edgecolors='black', zorder=10, label='Ear')
    ax_map.text(ear_x, 0.1, f"DRIVER'S EAR\n(Error Mic)", ha='center', fontweight='bold')

    # 3. Speaker (Anti-Noise)
    ax_map.scatter(spk_x, -0.2, s=300, c='green', edgecolors='black', marker='s', zorder=10, label='Speaker')
    ax_map.text(spk_x, -0.15, f"ANC SPEAKER", ha='center', fontweight='bold', color='green')

    # Draw Paths (Arrows)
    # Primary Path
    ax_map.annotate("", xy=(ear_x, 0), xytext=(src_x, 0), arrowprops=dict(arrowstyle="->", color='red', lw=2))
    ax_map.text(ear_x/2, 0.02, f"Primary Path P(z)\nDelay: {p_sample} samples\nDist: {p_dist*100:.1f} cm", ha='center', color='red', bbox=dict(facecolor='white', alpha=0.8))

    # Secondary Path
    ax_map.annotate("", xy=(ear_x, 0), xytext=(spk_x, -0.2), arrowprops=dict(arrowstyle="->", color='green', lw=2, linestyle='dashed'))
    ax_map.text((spk_x+ear_x)/2, -0.15, f"Secondary Path S(z)\nDelay: {s_sample} samples\nDist: {s_dist*100:.1f} cm", ha='center', color='green', bbox=dict(facecolor='white', alpha=0.8))

    # --- SUBPLOT 2: IMPULSE RESPONSES (Bottom Left) ---
    ax_p = fig.add_subplot(2, 2, 3)
    ax_p.stem(np.arange(len(p_ir)), p_ir, linefmt='r-', markerfmt='ro', basefmt='k-')
    ax_p.set_title("Primary Path P(z) - Impulse Response")
    ax_p.set_xlabel("Time (Samples)")
    ax_p.set_ylabel("Amplitude")
    ax_p.grid(True, alpha=0.3)
    ax_p.text(p_sample+2, max(p_ir)*0.9, "Pure Delay\n(Anechoic)", color='red')

    # --- SUBPLOT 3: IMPULSE RESPONSES (Bottom Right) ---
    ax_s = fig.add_subplot(2, 2, 4)
    ax_s.stem(np.arange(len(s_ir)), s_ir, linefmt='g-', markerfmt='go', basefmt='k-')
    ax_s.set_title("Secondary Path S(z) - Impulse Response")
    ax_s.set_xlabel("Time (Samples)")
    ax_s.grid(True, alpha=0.3)
    ax_s.text(s_sample+2, max(s_ir)*0.9, "Delay + Speaker Filter", color='green')

    # --- PARAMETER TABLE ---
    param_text = (
        f"SIMULATION PARAMETERS:\n"
        f"----------------------\n"
        f"Sample Rate:    {config.SAMPLE_RATE} Hz\n"
        f"Speed of Sound: 343 m/s\n"
        f"Engine Distance: {p_dist*100:.1f} cm\n"
        f"Speaker Distance:{s_dist*100:.1f} cm\n"
        f"Reflections:    NONE (Anechoic)"
    )
    fig.text(0.02, 0.9, param_text, fontsize=10, family='monospace', bbox=dict(facecolor='yellow', alpha=0.2))

    # Save
    out_dir = os.path.join(config.BASE_DIR, "results_plots")
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, "00_simulation_setup_diagram.png")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"Visualization generated: {save_path}")

if __name__ == "__main__":
    main()