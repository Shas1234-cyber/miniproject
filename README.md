Live Demo

👉 Deployed App: (https://traffic-ai-whjz.onrender.com)
Example:

https://your-app-name.onrender.com
📖 About the Project

This project is an AI-powered Traffic Prediction System that predicts traffic volume based on:

Time of day
Day of week
Weather conditions (live API)
City events

It uses Machine Learning (Random Forest Regressor) and provides a real-time interactive dashboard using Gradio.

🚀 Features
🤖 Machine Learning traffic prediction
🌦️ Live weather integration using API
📊 Smart traffic status (Low / Moderate / High)
🎯 City event impact analysis
🌐 Web-based interactive UI
⚡ Real-time predictions
🧠 Tech Stack
Python 🐍
Pandas
Scikit-learn
Random Forest Regressor
Gradio (UI)
OpenWeather API
Render (Deployment)
📂 Project Structure
miniproject/
│
├── app.py
├── traffic_dataset_with_trend.csv
├── requirements.txt
└── README.md
⚙️ How It Works
Dataset is loaded and processed
Features are extracted:
Hour
Day of Week
Weather
Events
Model is trained using Random Forest
User inputs city/time/event
Live weather is fetched via API
Model predicts traffic volume
