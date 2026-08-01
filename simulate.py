"""
simulate.py - Chopstick Crane headless simulation runner
=========================================================
Runs the full MuJoCo simulation, saves logs, optionally records video.
"""

import argparse
import time
import os

import mujoco
import numpy as np

from controller import ChopstickController

MODEL_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.xml")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# ─────────────────────────────────────────────────────────────────────────────
# Simulation runner
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(
    model_path:    str   = MODEL_PATH,
    results_dir:   str   = RESULTS_DIR,
    no_video:      bool  = True,
    video_fps:     int   = 60,
    render_width:  int   = 1280,
    render_height: int   = 720,
    cameras:       list  = None,
) -> dict:
    """
    Run the full simulation and return logs.

    Returns
    -------
    logs : dict  with keys t, theta, tip, phi, Fn, target, phase, s
    """
    os.makedirs(results_dir, exist_ok=True)

    print(f"[MuJoCo] Loading: {model_path}")
    model = mujoco.MjModel.from_xml_path(model_path)
    data  = mujoco.MjData(model)

    dt      = model.opt.timestep
    T_total = 40.0                # seconds (long run for stability demo)
    n_steps = int(T_total / dt)

    print(f"[Sim]    Duration: {T_total}s  ({n_steps} steps, dt={dt}s)")

    # Initialise controller
    ctrl = ChopstickController(model, data, dt=dt)
    
    # Initialize arm to home config so it doesn't spawn inside the board
    data.qpos[:3] = ctrl._theta_home
    mujoco.mj_forward(model, data)

    # Optional: set up renderer for video
    if cameras is None:
        cameras = ["iso_cam", "cam_front", "cam_top"]
    
    renderer = None
    frames   = {cam: [] for cam in cameras}
    if not no_video:
        try:
            renderer = mujoco.Renderer(model, height=render_height, width=render_width)
        except ValueError:
            print(f"[Warning] Framebuffer too small for {render_width}x{render_height}. Falling back to 640x480.")
            renderer = mujoco.Renderer(model, height=480, width=640)

    t_wall_start = time.time()
    report_every = int(1.0 / dt)  # every 1 second of sim time

    step = 0
    while not ctrl.done and step < n_steps:
        ctrl.step()
        mujoco.mj_step(model, data)

        if step % report_every == 0:
            logs_tmp = ctrl.get_logs()
            Fn_now   = logs_tmp["Fn"][-1]  if len(logs_tmp["Fn"])   > 0 else 0.0
            phi_now  = logs_tmp["phi"][-1] if len(logs_tmp["phi"])  > 0 else 0.0
            t_wall   = time.time() - t_wall_start
            print(f"  t={ctrl.sim_time:.1f}s  phase={ctrl.phase.name:10s}"
                  f"  Fn={Fn_now:.2f}N  phi={float(np.degrees(phi_now)):.1f} deg"
                  f"  [wall: {t_wall:.1f}s]")

        if renderer is not None and step % max(1, int(1.0 / (video_fps * dt))) == 0:
            for cam in cameras:
                renderer.update_scene(data, camera=cam)
                frames[cam].append(renderer.render().copy())

        step += 1

    t_wall = time.time() - t_wall_start
    print(f"[Sim]    Controller finished at step {step}")
    print(f"[Sim]    Finished {step} steps in {t_wall:.1f}s wall time")

    logs = ctrl.get_logs()

    # Save numpy arrays
    save_path = os.path.join(results_dir, "data.npz")
    np.savez(save_path, **{k: np.array(v) for k, v in logs.items()})
    print(f"[Data]   Saved logs -> {save_path}")

    # Save video
    if renderer is not None:
        try:
            import imageio
            name_map = {"iso_cam": "side", "cam_front": "front", "cam_top": "top"}
            for cam, f_list in frames.items():
                if f_list:
                    mapped_name = name_map.get(cam, cam)
                    vid_path = os.path.join(results_dir, f"simulation_{mapped_name}.mp4")
                    imageio.mimsave(vid_path, f_list, fps=video_fps)
                    print(f"[Video]  Saved -> {vid_path}")
        except ImportError:
            print("[Video]  imageio not installed; skipping video save.")
        renderer.close()

    return logs


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Chopstick Crane Simulation")
    parser.add_argument("--no-video",  action="store_true", help="Skip video recording")
    parser.add_argument("--fps",    type=int, default=60,   help="Video frame rate")
    parser.add_argument("--width",  type=int, default=1280, help="Video width")
    parser.add_argument("--height", type=int, default=720,  help="Video height")
    parser.add_argument("--cameras", nargs="+", default=["iso_cam", "cam_front", "cam_top"], help="Camera names to render from")
    args = parser.parse_args()

    logs = run_simulation(
        no_video      = args.no_video,
        video_fps     = args.fps,
        render_width  = args.width,
        render_height = args.height,
        cameras       = args.cameras,
    )

    # Quick summary
    if len(logs["Fn"]) > 0:
        fn_arr  = logs["Fn"]
        tip     = logs["tip"]
        target  = logs["target"]
        phi_arr = np.degrees(logs["phi"])
        phase   = logs["phase"]

        sweep = (phase == 2)

        print(f"\n-- Results Summary (SWEEP phase only) --")
        if sweep.sum() > 0:
            err = np.linalg.norm(tip[sweep] - target[sweep], axis=1)
            print(f"  Tracking error  mean={err.mean()*1000:.1f}mm  max={err.max()*1000:.1f}mm")
            print(f"  Contact force   mean={fn_arr[sweep].mean():.2f}N  "
                  f"min={fn_arr[sweep].min():.2f}N  max={fn_arr[sweep].max():.2f}N")
            in_band = 100.0 * np.mean((fn_arr[sweep] >= 1.5) & (fn_arr[sweep] <= 6.0))
            print(f"  Force in-band   {in_band:.1f}%")
        print(f"  Board tilt      mean={phi_arr.mean():.1f} deg  "
              f"range=[{phi_arr.min():.1f} deg, {phi_arr.max():.1f} deg]")
        print(f"\nRun plot_results.py to see plots.")


if __name__ == "__main__":
    main()
