"""Streamlit dashboard skeleton: pick a .npz sample, view the phase-folded
fit, a placeholder classification card, and a fitted-parameters card.

Streamlit reruns the whole script per interaction — expensive loads are
wrapped in @st.cache_data.
"""

import sys
from pathlib import Path

import numpy as np
import streamlit as st
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arvyo import contract
from arvyo.viz.plots import plot_fit

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


@st.cache_data
def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@st.cache_data
def list_samples(root_dir):
    root = Path(root_dir)
    if not root.exists():
        return []
    return sorted(str(p) for p in root.glob("*/*.npz"))


@st.cache_data
def load_sample_cached(path):
    return contract.load_sample(path)


def main():
    st.set_page_config(page_title="Arvyo", layout="wide")
    st.title("Arvyo — transit vetting dashboard")

    config = load_config()
    processed_root = config["data"]["processed_root"]

    st.sidebar.header("Sample")
    root_input = st.sidebar.text_input("Processed data root", processed_root)
    samples = list_samples(root_input)

    if not samples:
        st.warning(f"No .npz files found under {root_input}. "
                   "Point the sidebar at a valid processed_root.")
        return

    chosen = st.sidebar.selectbox("Sample", samples)

    try:
        sample = load_sample_cached(chosen)
    except contract.ContractError as exc:
        st.error(str(exc))
        return

    st.subheader(f"TIC {sample['tic_id']} — label: {sample['label']}")

    period = sample.get("period_days")
    epoch = sample.get("epoch_btjd")

    col_plot, col_cards = st.columns([3, 1])

    with col_cards:
        st.markdown("### Classification (placeholder)")
        st.write({"planet": 0.25, "eb": 0.25, "blend": 0.25, "starspot": 0.25})

        st.markdown("### Fitted parameters (placeholder)")
        st.write({
            "period_days": period,
            "epoch_btjd": epoch,
            "crowdsap": sample.get("crowdsap"),
        })

    with col_plot:
        if period and epoch and np.isfinite(period) and np.isfinite(epoch):
            model_flux = np.ones_like(sample["flux"])  # no fitted model yet
            fig = plot_fit(sample["time"], sample["flux"], model_flux, period, epoch)
            st.pyplot(fig)
        else:
            st.info("No period/epoch on this sample yet — nothing to phase-fold.")


if __name__ == "__main__":
    main()
