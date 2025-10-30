import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="KrowdDash", layout="wide")

DATA_DIR = "data"
csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]

st.sidebar.title("📁 Select Dataset")
selected_file = st.sidebar.selectbox("Choose a file", csv_files)

@st.cache_data
def load_data(file):
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, file))
        df["source"] = file.replace("_clean.csv", "")
        return df
    except Exception as e:
        st.error(f"Error loading {file}: {e}")
        return pd.DataFrame()

df = load_data(selected_file)

# Optional filters
if "City" in df.columns:
    cities = st.sidebar.multiselect("City", df["City"].unique())
    if cities:
        df = df[df["City"].isin(cities)]

if "Category" in df.columns:
    categories = st.sidebar.multiselect("Category", df["Category"].unique())
    if categories:
        df = df[df["Category"].isin(categories)]

st.title(f"KrowdDash: {selected_file.replace('_clean.csv','').title()}")
st.subheader("📊 Summary")
st.write(df.describe(include="all"))

st.subheader("📄 Data Table")
st.dataframe(df)

st.download_button("Download CSV", df.to_csv(index=False), "filtered.csv", "text/csv")