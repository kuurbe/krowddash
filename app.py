import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Krowd Guide Dashboard", layout="wide")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data/dallas_pilot_combined.csv")
    df["hour"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.hour
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("Filter")
selected_hour = st.sidebar.slider("Hour", 0, 23, (18, 23))
selected_type = st.sidebar.multiselect("Incident Type", df["type"].unique())

# Filter data
filtered = df[(df["hour"].between(*selected_hour))]
if selected_type:
    filtered = filtered[filtered["type"].isin(selected_type)]

# Hotspot map
st.subheader("📍 Hotspot Map")
fig = px.density_mapbox(filtered, lat="lat", lon="lon", z="severity",
                        radius=15, center=dict(lat=32.78, lon=-96.8),
                        zoom=12, mapbox_style="open-street-map")
st.plotly_chart(fig, use_container_width=True)

# Predictive model
st.subheader("🔮 Predictive Risk")
features = ["hour", "day_of_week", "weather_score", "event_score"]
X = filtered[features]
y = filtered["risk_level"]
model = RandomForestClassifier()
model.fit(X, y)
preds = model.predict(X)
filtered["predicted_risk"] = preds
st.dataframe(filtered[["timestamp", "type", "location", "predicted_risk"]])

# Export
st.download_button("Download Predictions", filtered.to_csv(index=False), "krowd_predictions.csv", "text/csv")