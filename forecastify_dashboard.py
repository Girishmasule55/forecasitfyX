import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from river.drift import ADWIN

st.set_page_config(page_title="Forecastify AI Dashboard", layout="wide")

st.title("📊 FORECASTIFY")
st.subheader("Smart Demand Forecasting with Concept Drift Detection")

st.markdown(
    """
Building adaptive AI systems that maintain accuracy in changing market conditions
through automatic drift detection and intelligent model retraining.
"""
)

uploaded_file = st.file_uploader(
    "Upload Transaction Dataset", type=["xlsx", "csv"])

if uploaded_file:

    if uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    st.sidebar.header("Dataset Preview")
    st.sidebar.write(df.head())

    st.header("Data Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", len(df))
    col2.metric("Total Columns", len(df.columns))
    col3.metric("Missing Values", df.isnull().sum().sum())

    st.dataframe(df.head())

    # ---- Auto Detect Date Column ----
    date_col = None
    for col in df.columns:
        if "date" in col.lower():
            date_col = col

    if date_col is None:
        st.error("Dataset must contain a date column")
    else:

        df[date_col] = pd.to_datetime(df[date_col])

        numeric_cols = df.select_dtypes(include=np.number).columns

        if len(numeric_cols) == 0:
            st.error("Dataset must contain numeric sales/demand column")
        else:

            demand_col = numeric_cols[0]

            df = df.sort_values(date_col)

            st.header("📈 Demand Trend")

            fig = px.line(df, x=date_col, y=demand_col,
                          title="Demand Over Time")
            st.plotly_chart(fig, use_container_width=True)

            # ----- Forecast Model -----

            st.header("🔮 Demand Forecast")

            df["time_index"] = np.arange(len(df))

            X = df[["time_index"]]
            y = df[demand_col]

            model = LinearRegression()
            model.fit(X, y)

            predictions = model.predict(X)

            df["Forecast"] = predictions

            fig2 = go.Figure()

            fig2.add_trace(go.Scatter(
                x=df[date_col],
                y=df[demand_col],
                mode='lines',
                name='Actual'
            ))

            fig2.add_trace(go.Scatter(
                x=df[date_col],
                y=df["Forecast"],
                mode='lines',
                name='Forecast'
            ))

            st.plotly_chart(fig2, use_container_width=True)

            # ---- Forecast Next 30 Days ----

            future_index = np.arange(len(df), len(df)+30).reshape(-1, 1)

            future_pred = model.predict(future_index)

            future_dates = pd.date_range(
                df[date_col].max(),
                periods=30,
                freq="D"
            )

            future_df = pd.DataFrame({
                "Date": future_dates,
                "Forecast": future_pred
            })

            fig3 = px.line(future_df, x="Date", y="Forecast",
                           title="Next 30 Day Forecast")
            st.plotly_chart(fig3, use_container_width=True)

            # ---- Drift Detection ----

            st.header("⚠ Concept Drift Detection")

            drift_detector = ADWIN()

            drift_points = []

            for i, val in enumerate(df[demand_col]):

                drift_detector.update(val)

                if drift_detector.drift_detected:
                    drift_points.append(i)

            drift_df = df.iloc[drift_points]

            fig4 = go.Figure()

            fig4.add_trace(go.Scatter(
                x=df[date_col],
                y=df[demand_col],
                mode='lines',
                name='Demand'
            ))

            fig4.add_trace(go.Scatter(
                x=drift_df[date_col],
                y=drift_df[demand_col],
                mode='markers',
                name='Drift Detected'
            ))

            st.plotly_chart(fig4, use_container_width=True)

            st.header("📊 Model Performance")

            error = mean_absolute_error(y, predictions)

            st.metric("Mean Absolute Error", round(error, 2))

            st.header("📉 Sales Distribution")

            fig5 = px.histogram(df, x=demand_col, nbins=40)
            st.plotly_chart(fig5, use_container_width=True)

            st.header("📊 Correlation Matrix")

            corr = df.corr(numeric_only=True)

            fig6 = px.imshow(corr, text_auto=True)
            st.plotly_chart(fig6, use_container_width=True)

            st.success("Forecastify Analysis Completed ✅")
