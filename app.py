import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="KrowdDash", layout="wide")

# 📂 Load available CSVs
DATA_DIR = "data"
csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]

st.sidebar.title("📁 Select Dataset")
selected_file = st.sidebar.selectbox("Choose a file to explore", csv_files)

# 📊 Load selected data
@st.cache_data
def load_data(file):
    return pd.read_csv(os.path.join(DATA_DIR, file))

df = load_data(selected_file)

# 🧪 Auto-detect filterable columns
filter_cols = [col for col in df.columns if df[col].nunique() < 50 and df[col].dtype == "object"]
st.sidebar.title("🔍 Filters")
for col in filter_cols:
    options = st.sidebar.multiselect(f"{col}", df[col].unique())
    if options:
        df = df[df[col].isin(options)]

# 📈 Dashboard
st.title(f"KrowdDash: {selected_file.replace('_clean.csv','').title()}")

# 🧮 Summary
st.subheader("📊 Summary Stats")
st.write(df.describe(include="all"))

# 📉 Chart Options
st.subheader("📈 Chart Builder")
numeric_cols = df.select_dtypes(include="number").columns.tolist()
categorical_cols = df.select_dtypes(include="object").columns.tolist()

x_axis = st.selectbox("X-axis", categorical_cols)
y_axis = st.selectbox("Y-axis", numeric_cols)

if x_axis and y_axis:
    chart = px.bar(df.groupby(x_axis)[y_axis].mean().reset_index(), x=x_axis, y=y_axis,
                   title=f"Average {y_axis} by {x_axis}")
    st.plotly_chart(chart, use_container_width=True)

# 📋 Data Table
st.subheader("📄 Filtered Data")
st.dataframe(df)

# 📥 Export
st.download_button("Download Filtered CSV", df.to_csv(index=False), "filtered_data.csv", "text/csv")

st.markdown("---")
st.caption("Built with ❤️ by Jacolby using Streamlit")