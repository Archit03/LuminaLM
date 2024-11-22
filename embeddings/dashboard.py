import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Dashboard Configuration
st.set_page_config(
    page_title="LuminaLM Training Dashboard",
    layout="wide",
)

# Paths
LOG_PATH = "training.log"
PLOTS_DIR = "plots"
METRICS_FILE = "metrics.csv"

# Load Training Logs
def load_logs(log_path):
    try:
        with open(log_path, "r") as f:
            logs = f.readlines()
        return logs
    except FileNotFoundError:
        return ["No logs available."]

# Load Metrics
def load_metrics(metrics_file):
    try:
        if os.path.exists(metrics_file):
            df = pd.read_csv(metrics_file)
            return df
        else:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading metrics: {e}")
        return pd.DataFrame()

# Load and Display Plots
def display_plots(plot_dir):
    plots = list(Path(plot_dir).glob("*.png"))
    if not plots:
        st.warning("No plots available.")
    else:
        for plot in plots:
            st.image(str(plot), caption=plot.name)

# Main Content
st.title("LuminaLM Training Dashboard")

# Logs Section
st.header("Training Logs")
logs = load_logs(LOG_PATH)
st.text_area("Logs", value="\n".join(logs[-100:]), height=300)

# Metrics Section
st.header("Training Metrics")
metrics_df = load_metrics(METRICS_FILE)
if not metrics_df.empty:
    st.dataframe(metrics_df)
    for column in metrics_df.columns[1:]:  # Skip the epoch column
        st.line_chart(metrics_df[["epoch", column]].set_index("epoch"))
else:
    st.warning("No metrics data available.")

# Visualizations Section
st.header("Visualizations")
display_plots(PLOTS_DIR)

# Footer
st.markdown("---")
st.markdown("Developed for LuminaLM Training Monitoring.")
