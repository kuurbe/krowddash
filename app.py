import streamlit as st
import pandas as pd
import os
import numpy as np
import plotly.express as px

st.set_page_config(page_title="KrowdDash", layout="wide")

# 📂 Load all CSVs from /data
DATA_DIR = "data"
csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]

@st.cache_data
def load_all_data():
    data_frames = []
    for file in csv_files:
        try:
            df = pd.read_csv(os.path.join(DATA_DIR, file))
            df["source"] = file.replace("_clean.csv", "")
            data_frames.append(df)
        except Exception as e:
            st.warning(f"Could not load {file}: {e}")
    return pd.concat(data_frames, ignore_index=True)

df = load_all_data()

# 🧪 Simulate foot traffic if missing
if "foot_traffic" not in df.columns:
    df["foot_traffic"] = np.random.randint(50, 500, size=len(df))

# 🧭 Sidebar filters
st.sidebar.header("🔍 Filter Data")
sources = st.sidebar.multiselect("Source", df["source"].unique())
cities = st.sidebar.multiselect("City", df["City"].unique() if "City" in df.columns else [])
hours = st.sidebar.slider("Hour of Day", 0, 23, (0, 23))

# 🧹 Apply filters
filtered_df = df.copy()
if sources:
    filtered_df = filtered_df[filtered_df["source"].isin(sources)]
if cities:
    filtered_df = filtered_df[filtered_df["City"].isin(cities)]
if "Timestamp" in filtered_df.columns:
    filtered_df["hour"] = pd.to_datetime(filtered_df["Timestamp"], errors="coerce").dt.hour
    filtered_df = filtered_df[(filtered_df["hour"] >= hours[0]) & (filtered_df["hour"] <= hours[1])]

# 📊 Summary Stats
st.title("📊 KrowdDash: All-in-One Insights")
st.subheader("Summary Statistics")
st.write(filtered_df.describe(include="all"))

# 🗺️ Plotly Heatmap (Scatter Map)
st.subheader("🗺️ Activity Map")
if "Latitude" in filtered_df.columns and "Longitude" in filtered_df.columns:
    fig = px.scatter_mapbox(
        filtered_df,
        lat="Latitude",
        lon="Longitude",
        color="source",
        size="foot_traffic",
        hover_name="source",
        zoom=10,
        height=600
    )
    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No location data found (Latitude/Longitude missing).")

# 📋 Data Table
st.subheader("📄 Filtered Data Table")
st.dataframe(filtered_df)

# 📥 Export
st.download_button("Download Filtered CSV", filtered_df.to_csv(index=False), "krowddash_filtered.csv", "text/csv")