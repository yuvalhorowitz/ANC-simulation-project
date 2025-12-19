# phase_3/simulation/test_run_scenario.py

import faulthandler
faulthandler.enable()

from phase_3.simulation.run_scenario import run_scenario


def main():
    print(">>> Starting test_run_scenario", flush=True)

    result = run_scenario(
        room_dims=(4.2, 3.1, 1.9),
        material_preset="mixed",
        noise_type="engine",
        noise_position=(1.0, 0.9, 1.0),
        ref_mic_position=(1.5, 1.1, 1.0),
        err_mic_position=(3.0, 2.3, 1.0),
        speaker_position=(2.8, 2.2, 1.0),
        duration=3.0,
        model_path="phase_2/models/tcn_ref2u.keras",
        anc_enabled=True,
        predict_batch_size=64,
    )

    print("\n=== METRICS ===")
    for k, v in result["metrics"].items():
        print(f"{k}: {v}")

    print("\n>>> Phase 3 scenario completed successfully.")
    print(">>> Exiting cleanly.")
    return 0


if __name__ == "__main__":
    main()
