# phase_3/visualization/plot_room.py

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def plot_room(geometry: dict):
    room_dims = geometry["room_dims"]
    noise_pos = geometry["noise_position"]
    ref_pos = geometry["ref_mic_position"]
    err_pos = geometry["err_mic_position"]
    spk_pos = geometry["speaker_position"]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Room box (just corners)
    L, W, H = room_dims
    corners = [
        (0, 0, 0), (L, 0, 0), (L, W, 0), (0, W, 0),
        (0, 0, H), (L, 0, H), (L, W, H), (0, W, H),
    ]
    xs, ys, zs = zip(*corners)
    ax.scatter(xs, ys, zs)

    # Entities
    ax.scatter([noise_pos[0]], [noise_pos[1]], [noise_pos[2]], marker="x", s=80, label="Noise Source")
    ax.scatter([spk_pos[0]], [spk_pos[1]], [spk_pos[2]], marker="^", s=80, label="Speaker")
    ax.scatter([ref_pos[0]], [ref_pos[1]], [ref_pos[2]], marker="o", s=80, label="Ref Mic")
    ax.scatter([err_pos[0]], [err_pos[1]], [err_pos[2]], marker="o", s=80, label="Err Mic")

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("Phase 3 Room Geometry")
    ax.legend()
    plt.show()
