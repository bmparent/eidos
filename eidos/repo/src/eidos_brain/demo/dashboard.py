"""
dashboard.py

Streamlit dashboard for visualizing Eidos Brain sessions.
Reads artifacts from local disk or GCS.
"""

import streamlit as st
import pandas as pd
import json
import os
import glob
from pathlib import Path

st.set_page_config(layout="wide", page_title="Eidos Brain Dashboard")

# Config
# Should be passed via env or args, but for Streamlit we often use env vars
ARTIFACT_ROOT = os.getenv("EIDOS_ARTIFACT_ROOT", "./eidos_artifacts_output")

st.title("Eidos Brain // Inspector")

@st.cache_data
def load_sessions(root: str):
    # Scan for manifest files
    manifests = glob.glob(os.path.join(root, "manifest_*.json"))
    sessions = []
    for m in manifests:
        try:
            with open(m, "r") as f:
                data = json.load(f)
                sessions.append(data)
        except Exception:
            pass
    return sessions

sessions = load_sessions(ARTIFACT_ROOT)

if not sessions:
    st.warning(f"No sessions found in {ARTIFACT_ROOT}")
    st.stop()

# Sidebar
st.sidebar.header("Session Inspector")
selected_session_meta = st.sidebar.selectbox(
    "Select Session", 
    sessions, 
    format_func=lambda s: f"{s['session_id']} ({s.get('ts', 'unknown')})"
)

if selected_session_meta:
    st.header(f"Session: {selected_session_meta['session_id']}")
    
    # Load session details
    # We expect a 'session_report.json' or similar if engine generated it.
    # Or we scan the artifacts folder for that session.
    # The 'manifest.jsonl' in the root logs all artifacts.
    
    # Let's try to load the 'manifest.jsonl' (global log) and filter for this session?
    # Or if the engine creates a per-session folder.
    # The engine code puts everything in ARTIFACT_ROOT directly or subdirs.
    
    # For this dashboard patch, we assume we want to see 'anomalies.csv' or similar if it exists.
    # The engine typically produces 'eidos_sentinel_report.txt' or similar.
    
    st.subheader("Configuration")
    st.json(selected_session_meta.get("config", {}))
    
    st.subheader("Provenance")
    col1, col2 = st.columns(2)
    col1.metric("Engine Hash", selected_session_meta.get("engine_hash")[:8])
    col2.metric("Config Hash", selected_session_meta.get("config_hash")[:8])
    
    # Placeholder for visualizations
    st.info("To see charts, ensure the engine outputs time-series CSVs (e.g. 'surprises.csv').")
    
    # If we had a CSV of timeline
    # scan for csvs in root/misc or root
    # For now, just show listing of artifacts associated
    
    st.subheader("Artifacts")
    # In a real impl, we'd use the references in the event log
    st.write("Artifacts would be listed here.")
    
