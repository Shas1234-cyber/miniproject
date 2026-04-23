import pandas as pd
import requests
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import gradio as gr

print("1. Loading Data and Training AI... Please wait.")

# Load dataset
df = pd.read_csv('traffic_dataset_with_trend.csv')

# Feature engineering
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Hour'] = df['Timestamp'].dt.hour
df['DayOfWeek'] = df['Timestamp'].dt.dayofweek

# Encoding
weather_encoder = LabelEncoder()
df['Weather_Encoded'] = weather_encoder.fit_transform(df['Weather'])
df['Events_Encoded'] = df['Events'].astype(int)

# Features & target
X = df[['Hour', 'DayOfWeek', 'Weather_Encoded', 'Events_Encoded']]
y = df['Traffic Volume']

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

print("✅ AI Trained Successfully! Launching Dashboard...")

# 🌦️ Live weather function
def get_live_weather(city):
    api_key = os.getenv("bfda9cfd2e90e13ff663d19e88a6018d")  # secure

    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}"
        response = requests.get(url, timeout=5)

        if response.status_code != 200:
            return "Clear"

        data = response.json()
        weather = data['weather'][0]['main']

        if weather.lower() in ["clear"]:
            return "Clear"
        elif weather.lower() in ["clouds"]:
            return "Cloudy"
        elif weather.lower() in ["rain", "drizzle"]:
            return "Rainy"
        else:
            return "Clear"

    except:
        return "Clear"


# 🚦 Prediction function
def predict_traffic(hour, day_name, city, is_event):
    day_map = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2,
        "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6
    }

    day_encoded = day_map[day_name]
    weather_name = get_live_weather(city)

    try:
        weather_encoded = weather_encoder.transform([weather_name])[0]
    except:
        weather_encoded = 0

    event_encoded = 1 if is_event == "Yes" else 0

    input_data = pd.DataFrame(
        [[hour, day_encoded, weather_encoded, event_encoded]],
        columns=['Hour', 'DayOfWeek', 'Weather_Encoded', 'Events_Encoded']
    )

    prediction = model.predict(input_data)[0]

    if prediction > 3000:
        status = "🔴 SEVERE CONGESTION"
        advice = "Avoid route. High probability of traffic jams and delays."
    elif prediction > 1500:
        status = "🟡 MODERATE TRAFFIC"
        advice = "Standard city flow. Minor slowdowns expected."
    else:
        status = "🟢 CLEAR ROADS"
        advice = "Smooth driving conditions. Optimal time to travel."

    return f"{int(prediction)} Vehicles/Hour", status, advice, weather_name


# 🎨 UI
with gr.Blocks() as modern_dashboard:
    gr.Markdown("<h1 style='text-align: center;'>🚦 Smart City Traffic Predictor AI</h1>")
    gr.Markdown("<h3 style='text-align: center; color: gray;'>SDG Goal 11: Sustainable Cities & Communities</h3>")

    with gr.Row():
        with gr.Column(scale=1, variant="panel"):
            gr.Markdown("### ⚙️ Input Parameters")

            hour_input = gr.Slider(0, 23, step=1, label="Time of Day", value=17)

            day_input = gr.Dropdown(
                ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                label="Day of the Week",
                value="Monday"
            )

            city_input = gr.Textbox(label="Enter City (e.g., Delhi)", value="Delhi")

            event_input = gr.Radio(["Yes", "No"], label="Is there a special city event today?", value="No")

            predict_btn = gr.Button("🔮 Predict Traffic Flow")

        with gr.Column(scale=1):
            gr.Markdown("### 📊 AI Real-Time Analysis")

            volume_output = gr.Textbox(label="Predicted Volume")
            status_output = gr.Textbox(label="Road Status Indicator")
            advice_output = gr.Textbox(label="AI Recommendation")
            weather_output = gr.Textbox(label="Live Weather")

    predict_btn.click(
        fn=predict_traffic,
        inputs=[hour_input, day_input, city_input, event_input],
        outputs=[volume_output, status_output, advice_output, weather_output]
    )


# 🚀 Smart launch (Colab + Render both)
if os.getenv("RENDER"):
    modern_dashboard.launch(
        server_name="0.0.0.0",
        server_port=10000
    )
else:
    modern_dashboard.launch()
