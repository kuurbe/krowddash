import streamlit as st
import pandas as pd
import os
import folium
from streamlit_folium import st_folium
import numpy as np

st.set_page_config(page_title="KrowdDash", layout="wide")

# 📂 Load all CSVs
DATA_DIR = "data"
csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]

@st.cache_data
def load_all_data():
    data_dict = {}
    for file in csv_files:
        df = pd.read_csv(os.path.join(DATA_DIR, file))
        df["source"] = file.replace("_clean.csv", "")
        data_dict[file] = df
    return pd.concat(data_dict.values(), ignore_index=True)

df_all = load_all_data()

# 🧪 Simulate foot traffic if missing
if "foot_traffic" not in df_all.columns:
    df_all["foot_traffic"] = np.random.randint(10, 500, size=len(df_all))

# 🗺️ Heatmap tab
st.title("🧭 All-in-One Insights: Heatmap + Stats")

# 🔍 Filters
with st.sidebar:
    st.header("Filter Options")
    selected_sources = st.multiselect("Data Sources", df_all["source"].unique())
    selected_city = st.multiselect("City", df_all["City"].unique() if "City" in df_all.columns else [])
    selected_hour = st.slider("Hour of Day", 0, 23, (0, 23))

# 🧹 Apply filters
filtered_df = df_all.copy()
if selected_sources:
    filtered_df = filtered_df[filtered_df["source"].isin(selected_sources)]
if selected_city:
    filtered_df = filtered_df[filtered_df["City"].isin(selected_city)]
if "Timestamp" in filtered_df.columns:
    filtered_df["hour"] = pd.to_datetime(filtered_df["Timestamp"], errors="coerce").dt.hour
    filtered_df = filtered_df[(filtered_df["hour"] >= selected_hour[0]) & (filtered_df["hour"] <= selected_hour[1])]

# 📊 Summary
st.subheader("📊 Summary Stats")
st.write(filtered_df.describe(include="all"))

# 🗺️ Heatmap
st.subheader("🗺️ Activity Heatmap")
if "Latitude" in filtered_df.columns and "Longitude" in filtered_df.columns:
    m = folium.Map(location=[filtered_df["Latitude"].mean(), filtered_df["Longitude"].mean()], zoom_start=12)
    for _, row in filtered_df.iterrows():
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=min(row["foot_traffic"] / 50, 10),
            popup=row["source"],
            color="blue",
            fill=True,
            fill_opacity=0.6
        ).add_to(m)
    st_folium(m, width=900, height=500)
else:
    st.warning("No location data found (Latitude/Longitude missing).")

# 📋 Data Table
st.subheader("📄 Filtered Data")
st.dataframe(filtered_df)

# 📥 Export
st.download_button("Download Filtered Data", filtered_df.to_csv(index=False), "all_insights.csv", "text/csv")