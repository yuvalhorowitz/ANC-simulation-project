import numpy as np
import pyroomacoustics as pra
import os
import random
import matplotlib.pyplot as plt

# --- הגדרות פרויקט ---
FS = 8000            # תדר דגימה אופטימלי לפי הלקחים
DURATION = 1.0       
DATA_DIR = "car_anc_final_data"
NUM_SCENARIOS = 10   

def plot_car_map_png(room_dim, src, ref, err, spks, scn_id):
    """מייצר מפת PNG של תא הנוסעים עם מקרא מפורט"""
    plt.figure(figsize=(10, 7))
    
    # גבולות הרכב
    plt.plot([0, room_dim[0], room_dim[0], 0, 0], 
             [0, 0, room_dim[1], room_dim[1], 0], 'k-', linewidth=2)

    # מיקומי רכיבים
    plt.scatter(src[0], src[1], marker='*', s=250, color='red', label='Engine (Source)')
    plt.scatter(ref[0], ref[1], marker='o', s=100, color='blue', label='Reference Mic')
    plt.scatter(err[0], err[1], marker='o', s=150, color='green', label='Driver Ear (Error Mic)')
    
    # רמקולים של הרכב (בדלתות)
    spk_array = np.array(spks)
    plt.scatter(spk_array[:, 0], spk_array[:, 1], marker='s', s=120, color='orange', label='Car Audio Speakers')

    plt.title(f"Car Cabin ANC Layout - Scenario {scn_id:03d}")
    plt.xlabel("Length (X) [m]"); plt.ylabel("Width (Y) [m]")
    plt.legend(loc='upper right', frameon=True, shadow=True)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.axis('equal')

    # שמירה כ-PNG
    png_path = os.path.join(DATA_DIR, f"map_scn_{scn_id:03d}.png")
    plt.savefig(png_path)
    plt.close()
    print(f"   ✓ Map saved: {png_path}")

def generate_engine_noise(duration, fs, base_f, intensity):
    """יצירת רעש מנוע מורכב עם הרמוניות"""
    t = np.linspace(0, duration, int(fs * duration))
    # הרמוניות מנוע (1 עד 5)
    signal = np.sum([(intensity/i) * np.sin(2*np.pi*base_f*i*t) for i in range(1, 6)], axis=0)
    # נרמול להבטחת יציבות האימון
    return (signal / np.max(np.abs(signal))).astype(np.float32)

def create_car_scenario(scn_id):
    if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)

    # 1. מימדי תא נוסעים (4.5 מטר אורך, 2.5 רוחב)
    room_dim = [4.5, 2.5, 1.5]
    abs_coeff = random.uniform(0.2, 0.4) # בליעה משתנה (ריפוד/חלונות)
    room = pra.ShoeBox(room_dim, fs=FS, max_order=4, absorption=abs_coeff)

    # 2. מיקומים פיזיקליים
    engine_pos = [0.5, 1.25, 0.8]
    ref_mic_pos = [0.7, 1.25, 0.8]
    
    # הזזה הדרגתית של הנהג (מושב זז קדימה/אחורה)
    driver_x = 2.4 + (scn_id * 0.05) 
    error_mic_pos = [driver_x, 0.75, 1.1] 
    
    # רמקולים בדלתות הקדמיות (רחוק מהאוזן)
    car_speakers = [[1.2, 0.2, 0.5], [1.2, 2.3, 0.5]]

    # 3. יצירת רעש ויזואליזציה
    noise = generate_engine_noise(DURATION, FS, random.uniform(45, 60), random.uniform(0.6, 1.0))
    room.add_source(engine_pos, signal=noise)
    room.add_microphone_array(pra.MicrophoneArray(np.array([ref_mic_pos, error_mic_pos]).T, FS))

    # הפקת המפה כ-PNG
    plot_car_map_png(room_dim, engine_pos, ref_mic_pos, error_mic_pos, car_speakers, scn_id)

    # 4. חישוב תגובת הלם (Hs) מהרמקולים לאוזן
    hs_rirs = []
    for spk in car_speakers:
        room_hs = pra.ShoeBox(room_dim, fs=FS, max_order=4, absorption=abs_coeff)
        room_hs.add_source(spk, signal=np.array([1.0] + [0.0]*255))
        room_hs.add_microphone(error_mic_pos)
        room_hs.compute_rir()
        hs_rirs.append(room_hs.rir[0][0][:256])
    
    # ממוצע של שני הרמקולים כ-Secondary Path אחד
    avg_hs_rir = np.mean(hs_rirs, axis=0).astype(np.float32)

    # סימולציה ושמירה
    room.simulate()
    prefix = os.path.join(DATA_DIR, f"scn_{scn_id:03d}")
    np.save(f"{prefix}_ref.npy", room.mic_array.signals[0]) # אות ייחוס
    np.save(f"{prefix}_mic.npy", room.mic_array.signals[1]) # אות המיקרופון באוזן
    np.save(f"{prefix}_hs.npy", avg_hs_rir)                 # תגובת ההלם של הרמקולים

if __name__ == "__main__":
    print(f"Generating {NUM_SCENARIOS} diverse car scenarios...")
    for i in range(NUM_SCENARIOS):
        create_car_scenario(i)
    print(f"\n✓ Process complete. Check '{DATA_DIR}' for .npy files and .png maps.")