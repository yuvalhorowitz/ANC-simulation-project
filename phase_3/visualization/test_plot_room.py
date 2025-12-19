from phase_3.simulation.run_scenario import run_scenario
from phase_3.visualization.plot_room import plot_room

if __name__ == "__main__":
    out = run_scenario(
        room_dims=(4, 3, 2),
        material_preset="mixed",
        noise_type="engine",
        noise_position=(1.0, 0.8, 1.0),
        listener_position=(3.0, 2.2, 1.0),
        speaker_position=(2.0, 1.5, 1.0),
        anc_enabled=False,
    )

    plot_room(out["geometry"])