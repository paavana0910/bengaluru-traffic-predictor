
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# -----------------------------
# 1. PAGE CONFIGURATION
# -----------------------------

st.set_page_config(
    page_title="Bengaluru Traffic Predictor",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 Bengaluru Traffic Congestion Predictor")
st.markdown(
    "### AI-powered traffic analysis and congestion prediction"
)

# -----------------------------
# 2. LOAD DATA AND MODEL
# -----------------------------

@st.cache_data
def load_data():
    return pd.read_csv("bengaluru_traffic.csv")

@st.cache_resource
def load_model():
    return joblib.load("best_traffic_model.pkl")

df = load_data()
model = load_model()

# -----------------------------
# 3. SIDEBAR
# -----------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose a section",
    [
        "Dashboard",
        "Traffic Analysis",
        "Predict Congestion"
    ]
)

# -----------------------------
# 4. DASHBOARD PAGE
# -----------------------------

if page == "Dashboard":

    st.header("Traffic Overview")

    total_vehicles = df["Total_Vehicles"].sum()
    avg_speed = df["Avg_Speed"].mean()
    locations = df["Location"].nunique()
    records = len(df)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Vehicles", f"{total_vehicles:,}")
    col2.metric("Average Speed", f"{avg_speed:.1f} km/h")
    col3.metric("Locations", locations)
    col4.metric("Total Records", records)

    st.divider()

    # Congestion distribution
    st.subheader("Congestion Level Distribution")

    congestion_counts = df["Congestion_Level"].value_counts().reset_index()
    congestion_counts.columns = ["Congestion", "Count"]

    fig1 = px.pie(
        congestion_counts,
        names="Congestion",
        values="Count",
        title="Distribution of Traffic Congestion",
        hole=0.4
    )

    st.plotly_chart(fig1, use_container_width=True)

    # Traffic by location
    st.subheader("Traffic by Location")

    location_traffic = (
        df.groupby("Location")["Total_Vehicles"]
        .mean()
        .reset_index()
        .sort_values("Total_Vehicles", ascending=False)
    )

    fig2 = px.bar(
        location_traffic,
        x="Location",
        y="Total_Vehicles",
        title="Average Number of Vehicles by Location",
        color="Total_Vehicles"
    )

    st.plotly_chart(fig2, use_container_width=True)

    # Map
    st.subheader("Bengaluru Traffic Locations")

    map_data = df[["Latitude", "Longitude"]].drop_duplicates()

# Rename columns for Streamlit map
    map_data = map_data.rename(columns={
    "Latitude": "latitude",
    "Longitude": "longitude"
})


    st.map(map_data)


# -----------------------------
# 5. TRAFFIC ANALYSIS PAGE
# -----------------------------

elif page == "Traffic Analysis":

    st.header("Traffic Analysis")

    # Traffic by hour
    st.subheader("Traffic by Time of Day")

    hourly_traffic = (
        df.groupby("Hour")["Total_Vehicles"]
        .mean()
        .reset_index()
    )

    fig3 = px.line(
        hourly_traffic,
        x="Hour",
        y="Total_Vehicles",
        markers=True,
        title="Average Traffic Volume by Hour"
    )

    st.plotly_chart(fig3, use_container_width=True)

    # Speed by location
    st.subheader("Average Vehicle Speed by Location")

    speed_data = (
        df.groupby("Location")["Avg_Speed"]
        .mean()
        .reset_index()
    )

    fig4 = px.bar(
        speed_data,
        x="Location",
        y="Avg_Speed",
        title="Average Speed Across Locations",
        color="Avg_Speed"
    )

    st.plotly_chart(fig4, use_container_width=True)

    # Weather impact
    st.subheader("Weather and Traffic")

    weather_data = (
        df.groupby("Weather")["Avg_Speed"]
        .mean()
        .reset_index()
    )

    fig5 = px.bar(
        weather_data,
        x="Weather",
        y="Avg_Speed",
        title="Average Speed Under Different Weather Conditions",
        color="Weather"
    )

    st.plotly_chart(fig5, use_container_width=True)

    # Raw data
    st.subheader("Traffic Dataset")

    st.dataframe(df, use_container_width=True)

# -----------------------------
# 6. PREDICTION PAGE
# -----------------------------

elif page == "Predict Congestion":

    st.header("Predict Next Traffic Congestion")

    st.write(
        "Enter the current traffic conditions and the next observation time "
        "to predict the upcoming congestion level."
    )

    with st.form("prediction_form"):

        location = st.selectbox(
            "Select Location",
            sorted(df["Location"].unique())
        )

        next_hour = st.selectbox(
            "Next Observation Hour",
            [8, 11, 17, 20]
        )

        next_day_type = st.selectbox(
            "Next Day Type",
            sorted(df["Day_Type"].unique())
        )

        previous_weather = st.selectbox(
            "Current Weather",
            sorted(df["Weather"].unique())
        )

        st.subheader("Previous Traffic Conditions")

        previous_total = st.number_input(
            "Previous Total Vehicles",
            min_value=0,
            max_value=5000,
            value=500
        )

        previous_cars = st.number_input(
            "Previous Cars",
            min_value=0,
            value=200
        )

        previous_bikes = st.number_input(
            "Previous Bikes",
            min_value=0,
            value=200
        )

        previous_buses = st.number_input(
            "Previous Buses",
            min_value=0,
            value=50
        )

        previous_autos = st.number_input(
            "Previous Autos",
            min_value=0,
            value=50
        )

        previous_speed = st.slider(
            "Previous Average Speed (km/h)",
            min_value=0.0,
            max_value=60.0,
            value=20.0
        )

        submitted = st.form_submit_button(
            "Predict Congestion"
        )

    if submitted:

        input_data = pd.DataFrame([{
            "Location": location,
            "Next_Hour": next_hour,
            "Next_Day_Type": next_day_type,
            "Previous_Total_Vehicles": previous_total,
            "Previous_Cars": previous_cars,
            "Previous_Bikes": previous_bikes,
            "Previous_Buses": previous_buses,
            "Previous_Autos": previous_autos,
            "Previous_Avg_Speed": previous_speed,
            "Previous_Weather": previous_weather
        }])

        prediction = model.predict(input_data)[0]

        st.divider()
        st.subheader("Prediction Result")

        if prediction == "High":
            st.error("🔴 HIGH CONGESTION EXPECTED")

        elif prediction == "Medium":
            st.warning("🟠 MEDIUM CONGESTION EXPECTED")

        else:
            st.success("🟢 LOW CONGESTION EXPECTED")

        st.write(
            f"Predicted congestion for **{location}** "
            f"at the next observation hour **{next_hour}:00**."
        )

        probabilities = model.predict_proba(input_data)[0]

        probability_data = pd.DataFrame({
            "Congestion Level": model.classes_,
            "Probability": probabilities
        })

        fig6 = px.bar(
            probability_data,
            x="Congestion Level",
            y="Probability",
            title="Prediction Probability",
            text_auto=".1%"
        )

        st.plotly_chart(fig6, use_container_width=True)

# -----------------------------
# 7. FOOTER
# -----------------------------

st.divider()

st.caption(
    "Bengaluru Traffic Congestion Predictor | "
    "Machine Learning Prototype | Synthetic Dataset"
)
