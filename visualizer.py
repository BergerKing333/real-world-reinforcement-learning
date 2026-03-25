#!/usr/bin/env python3
"""Catheter dataset visualizer — Streamlit app.

Usage:
    mamba run -n viz streamlit run visualizer.py -- /tmp/catheter_dataset
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Catheter Dataset Viewer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* tighter padding */
    .block-container { padding-top: 1rem; padding-bottom: 0rem; }
    /* metric cards */
    [data-testid="stMetric"] {
        background: #1e1e2e;
        border: 1px solid #313244;
        border-radius: 0.5rem;
        padding: 0.6rem 0.8rem;
    }
    [data-testid="stMetricValue"] { font-size: 1.3rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── helpers ──────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_dataset(jsonl_path: str) -> pd.DataFrame:
    """Load JSONL dataset into a flat DataFrame."""
    records = []
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            flat = {
                "trial": r["trial"],
                "step": r["step"],
                "trial_id": r["trial_id"],
                "timestamp": r["timestamp_iso"],
                "image_path": r["image_path"],
                "image_w": r["image_w"],
                "image_h": r["image_h"],
                # state before
                "ins_before_cm": r["state_before"]["insertion_cm"],
                "rot_before_rad": r["state_before"]["rotation_rad"],
                # command
                "cmd_ins": r["command"]["insertion"],
                "cmd_rot": r["command"]["rotation"],
                "cmd_rel": r["command"]["relative"],
                # state after
                "ins_after_cm": r["state_after"]["insertion_cm"],
                "rot_after_rad": r["state_after"]["rotation_rad"],
            }
            records.append(flat)
    df = pd.DataFrame(records)
    df.sort_values(["trial", "step"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def load_image(path: str, w: int, h: int) -> np.ndarray:
    """Load a raw BGR8 .bin file and return RGB uint8 array."""
    raw = np.fromfile(path, dtype=np.uint8)
    expected = h * w * 3
    if raw.size != expected:
        return np.zeros((h, w, 3), dtype=np.uint8)
    bgr = raw.reshape((h, w, 3))
    return bgr[:, :, ::-1]  # BGR → RGB


def downsample(img: np.ndarray, max_px: int = 1200) -> np.ndarray:
    """Down-sample for web display — simple slicing (fast, no OpenCV)."""
    h, w = img.shape[:2]
    factor = max(1, max(h, w) // max_px)
    if factor == 1:
        return img
    return img[::factor, ::factor].copy()


# ── colour palette ───────────────────────────────────────────────────────────
C_INS = "#89b4fa"   # blue
C_ROT = "#f9e2af"   # yellow
C_CMD = "#a6e3a1"   # green
C_MARK = "#f38ba8"  # red / current marker
C_GRID = "#45475a"

PLOT_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#1e1e2e",
    font=dict(family="Inter, sans-serif", size=12),
    margin=dict(l=50, r=20, t=35, b=40),
    xaxis=dict(gridcolor=C_GRID),
    yaxis=dict(gridcolor=C_GRID),
)


# ── resolve dataset path ────────────────────────────────────────────────────
def resolve_path() -> Path:
    # Accept from CLI args (after --)
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", nargs="?", default=None)
    args, _ = parser.parse_known_args()
    if args.dataset_dir:
        return Path(args.dataset_dir)
    # fallback
    return Path("/tmp/catheter_dataset")


dataset_dir = resolve_path()
jsonl_path = dataset_dir / "dataset.jsonl"

if not jsonl_path.exists():
    st.error(f"Dataset not found: {jsonl_path}")
    st.stop()

df = load_dataset(str(jsonl_path))
trials = sorted(df["trial"].unique())

# ── sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 Catheter Dataset")
    st.caption(f"{len(df)} records · {len(trials)} trials · {dataset_dir}")

    trial = st.selectbox("Trial", trials, format_func=lambda t: f"Trial {t}")
    trial_df = df[df["trial"] == trial].reset_index(drop=True)
    max_step = int(trial_df["step"].max())

    step = st.slider("Step", 0, max_step, 0, key="step_slider")
    row = trial_df[trial_df["step"] == step]
    if row.empty:
        st.warning(f"Step {step} missing in trial {trial}")
        st.stop()
    row = row.iloc[0]

    st.divider()
    st.subheader("State before")
    c1, c2 = st.columns(2)
    c1.metric("Insertion", f"{row.ins_before_cm:.2f} cm")
    c2.metric("Rotation", f"{row.rot_before_rad:.2f} rad")

    st.subheader("Command")
    c1, c2 = st.columns(2)
    c1.metric("Δ Ins", f"{row.cmd_ins:+.3f} cm")
    c2.metric("Δ Rot", f"{row.cmd_rot:+.3f} rad")

    st.subheader("State after")
    c1, c2 = st.columns(2)
    c1.metric("Insertion", f"{row.ins_after_cm:.2f} cm")
    c2.metric("Rotation", f"{row.rot_after_rad:.2f} rad")

    st.divider()
    show_trajectory = st.toggle("Trajectory overlay", value=True)
    show_polar = st.toggle("Polar view", value=False)

# ── main area ────────────────────────────────────────────────────────────────
# image
img_path = row.image_path
if Path(img_path).exists():
    rgb = load_image(img_path, int(row.image_w), int(row.image_h))
    rgb_small = downsample(rgb, max_px=1200)
else:
    rgb_small = None

col_img, col_charts = st.columns([1.2, 1], gap="large")

with col_img:
    st.markdown(f"### Trial {trial}  ·  Step {step}")
    if rgb_small is not None:
        st.image(rgb_small, width="stretch")
    else:
        st.warning("Image file not found")
    st.caption(f"`{img_path}`  ({row.image_w}×{row.image_h})")

with col_charts:
    # ── trajectory: insertion & rotation over steps ──────────────────────────
    if show_trajectory:
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=("Insertion (cm)", "Rotation (rad)"),
        )

        fig.add_trace(
            go.Scatter(
                x=trial_df["step"],
                y=trial_df["ins_after_cm"],
                mode="lines+markers",
                marker=dict(size=4),
                line=dict(color=C_INS, width=2),
                name="Insertion",
            ),
            row=1,
            col=1,
        )
        # current step marker
        fig.add_trace(
            go.Scatter(
                x=[step],
                y=[row.ins_after_cm],
                mode="markers",
                marker=dict(size=12, color=C_MARK, symbol="diamond"),
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=trial_df["step"],
                y=trial_df["rot_after_rad"],
                mode="lines+markers",
                marker=dict(size=4),
                line=dict(color=C_ROT, width=2),
                name="Rotation",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[step],
                y=[row.rot_after_rad],
                mode="markers",
                marker=dict(size=12, color=C_MARK, symbol="diamond"),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        fig.update_layout(
            **PLOT_LAYOUT,
            height=420,
            showlegend=False,
        )
        fig.update_xaxes(title_text="Step", row=2, col=1, gridcolor=C_GRID)
        fig.update_yaxes(gridcolor=C_GRID)
        st.plotly_chart(fig, width="stretch")

    # ── polar / phase plot ───────────────────────────────────────────────────
    if show_polar:
        fig_polar = go.Figure()

        # trail
        fig_polar.add_trace(
            go.Scatterpolar(
                r=trial_df["ins_after_cm"],
                theta=trial_df["rot_after_rad"].apply(math.degrees),
                mode="lines+markers",
                marker=dict(size=4, color=C_INS),
                line=dict(color=C_INS, width=1.5),
                name="Path",
            )
        )
        # current
        fig_polar.add_trace(
            go.Scatterpolar(
                r=[row.ins_after_cm],
                theta=[math.degrees(row.rot_after_rad)],
                mode="markers",
                marker=dict(size=14, color=C_MARK, symbol="diamond"),
                name="Current",
            )
        )

        fig_polar.update_layout(
            **{k: v for k, v in PLOT_LAYOUT.items() if k != "xaxis" and k != "yaxis"},
            height=380,
            polar=dict(
                bgcolor="#1e1e2e",
                radialaxis=dict(gridcolor=C_GRID, color="#cdd6f4"),
                angularaxis=dict(gridcolor=C_GRID, color="#cdd6f4"),
            ),
            showlegend=False,
        )
        st.plotly_chart(fig_polar, width="stretch")

    # ── command bar chart ────────────────────────────────────────────────────
    st.markdown("#### Commands this trial")
    fig_cmd = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Δ Insertion (cm)", "Δ Rotation (rad)"),
    )
    fig_cmd.add_trace(
        go.Bar(
            x=trial_df["step"],
            y=trial_df["cmd_ins"],
            marker_color=C_INS,
            name="Δ Ins",
        ),
        row=1,
        col=1,
    )
    fig_cmd.add_trace(
        go.Bar(
            x=trial_df["step"],
            y=trial_df["cmd_rot"],
            marker_color=C_ROT,
            name="Δ Rot",
        ),
        row=2,
        col=1,
    )
    # highlight current bar
    for r_idx, col_color, field in [(1, C_MARK, "cmd_ins"), (2, C_MARK, "cmd_rot")]:
        fig_cmd.add_trace(
            go.Bar(
                x=[step],
                y=[row[field]],
                marker_color=col_color,
                showlegend=False,
                width=0.9,
            ),
            row=r_idx,
            col=1,
        )

    fig_cmd.update_layout(
        **PLOT_LAYOUT,
        height=350,
        showlegend=False,
        bargap=0.15,
    )
    fig_cmd.update_xaxes(title_text="Step", row=2, col=1, gridcolor=C_GRID)
    fig_cmd.update_yaxes(gridcolor=C_GRID)
    st.plotly_chart(fig_cmd, width="stretch")

# ── footer ───────────────────────────────────────────────────────────────────
st.divider()
with st.expander("Raw record JSON"):
    st.json(row.to_dict())
