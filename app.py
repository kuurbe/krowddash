import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import plotly.express as px

st.set_page_config(page_title="KrowdDash Hotspots", layout="wide")

# 📂 Load CSVs
DATA_DIR = "data"
csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]

@st.cache_data
def load_all_data():
    frames = []
    for file in csv_files:
        try:
            df = pd.read_csv(os.path.join(DATA_DIR, file))
            df["source"] = file.replace("_clean.csv", "")
            frames.append(df)
        except:
            continue
    return pd.concat(frames, ignore_index=True)

df = load_all_data()

# 🧪 Simulate foot traffic if missing
if "foot_traffic" not in df.columns:
    df["foot_traffic"] = np.random.randint(50, 500, size=len(df))

# 🧭 Sidebar filters
st.sidebar.title("🎛️ Explore")
sources = st.sidebar.multiselect("Source", df["source"].unique())
cities = st.sidebar.multiselect("City", df["City"].unique() if "City" in df.columns else [])
hour_range = st.sidebar.slider("Hour of Day", 0, 23, (0, 23))

# 🧹 Apply filters
filtered_df = df.copy()
if sources:
    filtered_df = filtered_df[filtered_df["source"].isin(sources)]
if cities:
    filtered_df = filtered_df[filtered_df["City"].isin(cities)]
if "Timestamp" in filtered_df.columns:
    filtered_df["hour"] = pd.to_datetime(filtered_df["Timestamp"], errors="coerce").dt.hour
    filtered_df = filtered_df[(filtered_df["hour"] >= hour_range[0]) & (filtered_df["hour"] <= hour_range[1])]

# 📊 Summary
st.title("📍 KrowdDash: Hotspots & Foot Traffic")
st.subheader("📊 Summary Stats")
st.write(filtered_df.describe(include="all"))

# 🔥 Hotspot Detection
st.subheader("🔥 Hotspot Map")
if "Latitude" in filtered_df.columns and "Longitude" in filtered_df.columns:
    coords = filtered_df[["Latitude", "Longitude"]].dropna()
    clustering = DBSCAN(eps=0.01, min_samples=5).fit(coords)
    filtered_df["hotspot"] = clustering.labels_

    fig = px.scatter_mapbox(
        filtered_df,
        lat="Latitude",
        lon="Longitude",
        color="hotspot",
        size="foot_traffic",
        hover_name="source",
        zoom=11,
        height=600
    )
    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No location data found.")

# 🧠 Predictive Widget
st.subheader("🔮 Predict Foot Traffic")
numeric_cols = filtered_df.select_dtypes(include=np.number).columns.tolist()
if "foot_traffic" in numeric_cols:
    numeric_cols.remove("foot_traffic")

features = st.multiselect("Select features to train model", numeric_cols)
if features:
    X = filtered_df[features]
    y = filtered_df["foot_traffic"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    st.write("📈 Sample Predictions")
    st.dataframe(pd.DataFrame({
        "Actual": y_test.values[:10],
        "Predicted": predictions[:10]
    }))
else:
    st.info("Select features to train the model.")

# 📥 Export
st.download_button("Download Filtered Data", filtered_df.to_csv(index=False), "krowddash_filtered.csv", "text/csv")

st.markdown("---")
st.caption("Built with ❤️ by Jacolby using Streamlit")