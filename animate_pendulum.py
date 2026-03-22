import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba

RESULTS_DIR = "results"
TRAIL_LENGTH = 250
STEP = 5
FPS = 60
SAVE_FILE = "caos.mp4" 
TARGET_FRAME = None


def load_trajectories(results_dir: str) -> list[pd.DataFrame]:
    pattern = os.path.join(results_dir, "*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[ERROR] No CSVs found in '{results_dir}'")
        sys.exit(1)
    print(f"[INFO] Loading {len(files)} trajectories from '{results_dir}'...")
    trajs = [pd.read_csv(f) for f in files]
    print(f"       {len(trajs[0])} frames, {len(trajs)} trajectories")
    return trajs


def infer_n_segments(df: pd.DataFrame) -> int:
    return sum(1 for c in df.columns if c.startswith("q") and c[1:].isdigit())


def angles_to_cartesian(thetas: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    n = len(thetas)
    pos = np.zeros((n + 1, 2))
    for i in range(n):
        pos[i + 1, 0] = pos[i, 0] + lengths[i] * np.sin(thetas[i])
        pos[i + 1, 1] = pos[i, 1] - lengths[i] * np.cos(thetas[i])
    return pos


def plot_single_frame(trajs: list[pd.DataFrame],
                      lengths: np.ndarray,
                      trail_len: int,
                      step: int,
                      frame_target: int) -> plt.Figure:
    
    n = infer_n_segments(trajs[0])
    q_cols = [f"q{i}" for i in range(n)]
    n_traj = len(trajs)

    all_thetas = [traj[q_cols].values for traj in trajs]
    all_times  = trajs[0]["t"].values

    frame_target = min(max(0, frame_target), len(all_times) - 1)
    t = all_times[frame_target]

    total_len = lengths.sum()

    fig = plt.figure(figsize=(10.8, 19.2), facecolor="#0D0D0F")
    ax  = fig.add_axes([0, 0, 1, 1], facecolor="#0D0D0F")
    ax.set_aspect("equal")

    margin = total_len * 1.05
    ax.set_xlim(-margin, margin)

    y_height = (2 * margin) * (1920 / 1080)
    y_top = margin * 0.5
    y_bottom = y_top - y_height
    ax.set_ylim(y_bottom, y_top)
    ax.axis("off")

    theta_bg = np.linspace(0, 2 * np.pi, 300)
    ax.plot(total_len * np.cos(theta_bg),
            total_len * np.sin(theta_bg),
            color="white", alpha=0.04, lw=0.5, zorder=0)

    ax.plot(0, 0, "o", color="white", ms=5, zorder=10, alpha=0.7)

    ax.text(
        0.05, 0.94, f"t = {t:6.2f} s", transform=ax.transAxes,
        color="white", alpha=0.5, fontsize=30,
        fontfamily="monospace", va="top"
    )

    ax.text(
        0.05, 0.98,
        f"{n}-segment pendulum  —  chaotic divergence",
        transform=ax.transAxes,
        color="white", alpha=0.35, fontsize=30, va="top"
    )

    cmap = plt.get_cmap("hsv")
    dynamic_colors = cmap(np.linspace(0, 1, n_traj))

    start_idx = max(0, frame_target - (trail_len * step))
    history_indices = np.arange(start_idx, frame_target + 1, step)
    
    if history_indices[-1] != frame_target:
        history_indices = np.append(history_indices, frame_target)

    for t_idx in range(n_traj):
        color = dynamic_colors[t_idx]
        rgba = to_rgba(color)

        pos_history = []
        for idx in history_indices:
            thetas = all_thetas[t_idx][idx]
            pos_history.append(angles_to_cartesian(thetas, lengths)[-1])
        pos_history = np.array(pos_history)

        if len(pos_history) > 1:
            pts = pos_history.reshape(-1, 1, 2)
            segments = np.concatenate([pts[:-1], pts[1:]], axis=1)
            n_seg = len(segments)
            alphas = np.linspace(0.0, 0.7, n_seg)
            colors_arr = np.array([(*rgba[:3], a) for a in alphas])
            
            lc = LineCollection(segments, colors=colors_arr, linewidths=0.8, zorder=3)
            ax.add_collection(lc)

        current_thetas = all_thetas[t_idx][frame_target]
        pos = angles_to_cartesian(current_thetas, lengths)

        for seg_i in range(n):
            ax.plot([pos[seg_i, 0], pos[seg_i + 1, 0]],
                    [pos[seg_i, 1], pos[seg_i + 1, 1]],
                    "-", color=color, lw=1.6, alpha=0.85, zorder=5)

        ax.plot([pos[-1, 0]], [pos[-1, 1]], "o", color=color, ms=5, alpha=0.95, zorder=6)

    return fig


def build_animation(trajs: list[pd.DataFrame],
                    lengths: np.ndarray,
                    trail_len: int,
                    interval: int,
                    step: int) -> tuple[plt.Figure, animation.FuncAnimation]:

    n = infer_n_segments(trajs[0])
    q_cols = [f"q{i}" for i in range(n)]
    n_traj = len(trajs)

    all_thetas = [traj[q_cols].values for traj in trajs]
    all_times  = trajs[0]["t"].values

    frames_idx = np.arange(0, len(all_times), step)
    n_frames   = len(frames_idx)

    total_len = lengths.sum()

    fig = plt.figure(figsize=(10.8, 19.2), facecolor="#0D0D0F")
    ax  = fig.add_axes([0, 0, 1, 1], facecolor="#0D0D0F")
    ax.set_aspect("equal")
    
    margin = total_len * 1.05
    ax.set_xlim(-margin, margin)

    y_height = (2 * margin) * (1920 / 1080)
    y_top = total_len * 1.15
    y_bottom = y_top - y_height
    ax.set_ylim(y_bottom, y_top)
    ax.axis("off")

    theta_bg = np.linspace(0, 2 * np.pi, 300)
    ax.plot(total_len * np.cos(theta_bg),
            total_len * np.sin(theta_bg),
            color="white", alpha=0.04, lw=0.5, zorder=0)

    ax.plot(0, 0, "o", color="white", ms=5, zorder=10, alpha=0.7)

    time_text = ax.text(
        0.05, 0.94, "", transform=ax.transAxes,
        color="white", alpha=0.5, fontsize=30,
        fontfamily="monospace", va="top"
    )

    ax.text(
        0.05, 0.98,
        f"{n}-segment pendulum  —  chaotic divergence",
        transform=ax.transAxes,
        color="white", alpha=0.35, fontsize=30, va="top"
    )

    rod_lines  = []
    mass_dots  = []
    trail_segs = []

    cmap = plt.get_cmap("hsv")
    dynamic_colors = cmap(np.linspace(0, 1, n_traj))

    for t_idx in range(n_traj):
        color = dynamic_colors[t_idx]

        segs = []
        for _ in range(n):
            ln, = ax.plot([], [], "-", color=color,
                          lw=1.6, alpha=0.85, zorder=5)
            segs.append(ln)
        rod_lines.append(segs)

        dot, = ax.plot([], [], "o", color=color,
                       ms=5, alpha=0.95, zorder=6)
        mass_dots.append(dot)

        rgba = to_rgba(color)
        lc = LineCollection([], linewidths=0.8, zorder=3)
        ax.add_collection(lc)
        trail_segs.append((lc, rgba))

    trail_buffer = [[] for _ in range(n_traj)]

    def init():
        for segs in rod_lines:
            for ln in segs:
                ln.set_data([], [])
        for dot in mass_dots:
            dot.set_data([], [])
        for lc, _ in trail_segs:
            lc.set_segments([])
        time_text.set_text("")
        return []

    def update(frame_num):
        fi = frames_idx[frame_num]
        t  = all_times[fi]
        time_text.set_text(f"t = {t:6.2f} s")

        for t_idx in range(n_traj):
            thetas = all_thetas[t_idx][fi]
            pos    = angles_to_cartesian(thetas, lengths)

            for seg_i, ln in enumerate(rod_lines[t_idx]):
                ln.set_data([pos[seg_i, 0], pos[seg_i + 1, 0]],
                            [pos[seg_i, 1], pos[seg_i + 1, 1]])

            mass_dots[t_idx].set_data([pos[-1, 0]], [pos[-1, 1]])

            trail_buffer[t_idx].append(pos[-1].copy())
            if len(trail_buffer[t_idx]) > trail_len:
                trail_buffer[t_idx].pop(0)

            buf = np.array(trail_buffer[t_idx])
            lc, rgba = trail_segs[t_idx]

            if len(buf) > 1:
                pts = buf.reshape(-1, 1, 2)
                segments = np.concatenate([pts[:-1], pts[1:]], axis=1)
                n_seg = len(segments)
                alphas = np.linspace(0.0, 0.7, n_seg)
                colors_arr = np.array([(*rgba[:3], a) for a in alphas])
                lc.set_segments(segments)
                lc.set_colors(colors_arr)
            else:
                lc.set_segments([])

        return []

    anim = animation.FuncAnimation(
        fig, update,
        frames=n_frames,
        init_func=init,
        interval=interval,
        blit=False
    )

    return fig, anim


def main():
    trajs = load_trajectories(RESULTS_DIR)
    n = infer_n_segments(trajs[0])
    lengths = np.full(n, 0.5)

    if TARGET_FRAME is not None:
        print(f"[INFO] Rendering single frame: {TARGET_FRAME}")
        fig = plot_single_frame(
            trajs=trajs,
            lengths=lengths,
            trail_len=TRAIL_LENGTH,
            step=STEP,
            frame_target=TARGET_FRAME
        )
        if SAVE_FILE:
            print(f"[INFO] Saving frame to '{SAVE_FILE}'...")
            fig.savefig(SAVE_FILE, dpi=100, bbox_inches="tight", pad_inches=0)
            print("[INFO] Done.")
        else:
            plt.show()
    else:
        interval = int(1000 / FPS)
        fig, anim = build_animation(
            trajs=trajs,
            lengths=lengths,
            trail_len=TRAIL_LENGTH,
            interval=interval,
            step=STEP,
        )

        if SAVE_FILE:
            print(f"[INFO] Saving animation to '{SAVE_FILE}'...")
            ext = os.path.splitext(SAVE_FILE)[1].lower()
            if ext == ".gif":
                writer = animation.PillowWriter(fps=FPS)
            else:
                writer = animation.FFMpegWriter(fps=FPS, bitrate=2000,
                                                extra_args=["-vcodec", "libx264"])
            anim.save(SAVE_FILE, writer=writer, dpi=100)
            print("[INFO] Done.")
        else:
            plt.show()


if __name__ == "__main__":
    main()