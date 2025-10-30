import streamlit as st
import pandas as pd
import os
import numpy as np
import folium
from streamlit_folium import st_folium

# 🔧 Page setup
st.set_page_config(page_title="KrowdDash Insights", layout="wide")

# 📂 Load all CSVs from /data
DATA_DIR = "data"
csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]

@st.cache_data
def load_all_data():
    data_frames = []
    for file in csv_files:
        df = pd.read_csv(os.path.join(DATA_DIR, file))
        df["source"] = file.replace("_clean.csv", "")
        data_frames.append(df)
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

# 🗺️ Heatmap
st.subheader("🗺️ Activity Heatmap")
if "Latitude" in filtered_df.columns and "Longitude" in filtered_df.columns:
    m = folium.Map(location=[filtered_df["Latitude"].mean(), filtered_df["Longitude"].mean()], zoom_start=12)
    for _, row in filtered_df.iterrows():
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=min(row["foot_traffic"] / 50, 10),
            popup=f"{row['source']}",
            color="blue",
            fill=True,
            fill_opacity=0.6
        ).add_to(m)
    st_folium(m, width=900, height=500)
else:
    st.warning("No location data found (Latitude/Longitude missing).")

# 📋 Data Table
st.subheader("📄 Filtered Data Table")
st.dataframe(filtered_df)

# 📥 Export
st.download_button("Download Filtered CSV", filtered_df.to_csv(index=False), "krowddash_filtered.csv", "text/csv")

# 🧾 Footer
st.markdown("---")
st.caption("Built with ❤️ by Jacolby using Streamlit")