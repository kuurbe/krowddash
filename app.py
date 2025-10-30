import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="KrowdDash Predictive Dashboard", layout="wide")

# 📂 Load CSVs from /data
DATA_DIR = "data"
csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]

st.sidebar.title("📁 Select Dataset")
selected_file = st.sidebar.selectbox("Choose a file", csv_files)

@st.cache_data
def load_data(file):
    df = pd.read_csv(os.path.join(DATA_DIR, file))
    df["source"] = file.replace("_clean.csv", "")
    return df

df = load_data(selected_file)

st.title(f"🔮 Predictive Dashboard: {selected_file.replace('_clean.csv','').title()}")

# 🧪 Simulate target column if missing
if "foot_traffic" not in df.columns:
    df["foot_traffic"] = np.random.randint(50, 500, size=len(df))

# 📊 Show raw data
st.subheader("📄 Raw Data")
st.dataframe(df)

# 🧠 Select features and target
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
if "foot_traffic" in numeric_cols:
    numeric_cols.remove("foot_traffic")

st.sidebar.header("🔧 Model Settings")
features = st.sidebar.multiselect("Select features", numeric_cols)
target = "foot_traffic"

if features:
    X = df[features]
    y = df[target]

    # 🧪 Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 🚀 Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 🔍 Predict
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    st.subheader("📈 Prediction Results")
    st.write(f"Mean Squared Error: {mse:.2f}")
    st.write("Sample Predictions:")
    st.dataframe(pd.DataFrame({
        "Actual": y_test.values[:10],
        "Predicted": predictions[:10]
    }))

    # 📥 Export predictions
    export_df = X_test.copy()
    export_df["Actual"] = y_test
    export_df["Predicted"] = predictions
    st.download_button("Download Predictions", export_df.to_csv(index=False), "predictions.csv", "text/csv")
else:
    st.info("Select at least one feature to train the model.")