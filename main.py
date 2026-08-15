import os
import json
import time
import requests
import telebot
from typing import Optional
from pydantic import BaseModel, Field
from google import genai
from FlightRadarAPI import FlightRadar24API

# ---------------------------------------------------------
# 1. Configuration & Client Initialization
# ---------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN) if TELEGRAM_BOT_TOKEN else None
client = genai.Client(api_key=GEMINI_API_KEY)
fr_api = FlightRadar24API()

# ---------------------------------------------------------
# 2. Pydantic Schemas for AI Output
# ---------------------------------------------------------
class FlightReport(BaseModel):
    """Schema for interactive chatbot responses."""
    flight_id: str = Field(description="Flight callsign or identifier")
    status: str = Field(description="ON_TIME, DELAYED, CANCELLED, or UNKNOWN")
    delay_probability_percent: int = Field(description="0 to 100 percentage likelihood of delay")
    predicted_delay_minutes: int = Field(description="Estimated delay duration in minutes")
    schedule_summary: str = Field(description="Departure, Arrival, and Time breakdown")
    weather_summary: str = Field(description="Summary of weather at origin or destination airport")
    detailed_reasons: str = Field(description="In-depth explanation of delay causes or safety hazards. Explicitly analyze weather, ATC holds, maintenance, and airspace restrictions like drone or unidentified aircraft sightings.")

class AutomatedAlert(BaseModel):
    """Schema for 24/7 background airspace monitor alerts."""
    has_risk: bool = Field(description="True if severe weather or dangerous delay risk is detected")
    severity: str = Field(description="LOW, MEDIUM, HIGH, or CRITICAL")
    flight_id: str = Field(description="Flight callsign affected")
    risk_factor: str = Field(description="Short hazard title")
    predicted_delay_minutes: int = Field(description="Estimated delay duration")
    recommended_action: str = Field(description="Actionable advice for flight operations")

# ---------------------------------------------------------
# 3. Data Source Helpers (OpenSky, FlightRadar24, Open-Meteo)
# ---------------------------------------------------------
def get_specific_flight(flight_number: str):
    """Fetches real-time radar details for a given flight number via FlightRadar24."""
    try:
        clean_no = flight_number.upper().strip().replace(" ", "")
        flights = fr_api.get_flights(flight_name=clean_no)
        if not flights:
            return None
        return fr_api.get_flight_details(flights[0])
    except Exception as e:
        print(f"FlightRadar24 fetch error: {e}")
        return None

def get_sector_flights():
    """Fetches active flights in a bounding box via OpenSky Network (No key required)."""
    try:
        url = "https://opensky-network.org/api/states/all?lamin=10.0&lomin=100.0&lamax=20.0&lomax=110.0"
        res = requests.get(url, timeout=10)
        states = res.json().get("states", [])
        return states[:5] if states else []
    except Exception:
        return []

def get_airport_weather(lat: float, lon: float):
    """Fetches real-time weather from Open-Meteo (No key required)."""
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,wind_speed_10m,weather_code,precipitation&hourly=visibility,wind_gusts_10m"
        res = requests.get(url, timeout=10)
        return res.json().get("current", {})
    except Exception:
        return {}

# ---------------------------------------------------------
# 4. AI Analysis via Gemini 2.5 Flash
# ---------------------------------------------------------
def analyze_interactive_query(flight_data, weather_data, flight_num: str) -> FlightReport:
    prompt = f"""
    You are an expert aviation operations AI.
    A user requested status for Flight: {flight_num}
    
    Live Radar Telemetry: {json.dumps(flight_data if flight_data else "NO ACTIVE RADAR SIGNAL. The plane is currently grounded at gate, between flights, or transponder is off.")}
    Live Weather Data: {json.dumps(weather_data)}
    
    Tasks:
    1. If live radar telemetry is missing, state clearly in schedule_summary that the flight is currently not airborne / on ground. Do not invent arbitrary departure times.
    2. Assess weather hazards for the provided airport coordinates.
    3. Predict delay likelihood (0-100%) and estimated delay minutes.
    4. Provide detailed disruption reasoning (ATC, ground holds, inbound aircraft delays, weather).
    """
    
    response = client.models.generate_content(
        model='gemini-1.5-flash', # Or gemini-2.0-flash
        contents=prompt,
        config={
            'response_mime_type': 'application/json',
            'response_schema': FlightReport,
        },
    )
    return response.parsed

def analyze_sector_monitor(flights, weather) -> AutomatedAlert:
    """Analyzes background sector telemetry for automated push alerts."""
    prompt = f"Analyze live sector flight telemetry: {json.dumps(flights)} and weather: {json.dumps(weather)}. Detect critical delay or safety risks."
    response = client.models.generate_content(
        model='gemini-3.6-flash',
        contents=prompt,
        config={
            'response_mime_type': 'application/json',
            'response_schema': AutomatedAlert,
        },
    )
    return response.parsed

# ---------------------------------------------------------
# 5. Telegram Handlers & Dispatchers
# ---------------------------------------------------------
@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    welcome_text = (
        "✈️ **Aviation Monitoring Chatbot**\n\n"
        "Send me any **Flight Number** (e.g., `AA100`, `VN123`, `DL456`).\n\n"
        "I will analyze live radar coordinates, weather forecasts, delay probabilities, "
        "and explain root causes (such as storms, ATC holds, or drone disruptions)!"
    )
    bot.reply_to(message, welcome_text, parse_mode="Markdown")

@bot.message_handler(func=lambda message: True)
@bot.message_handler(func=lambda message: True)
def handle_flight_search(message):
    flight_num = message.text.strip().upper()
    status_msg = bot.reply_to(message, f"🔍 Querying live radar and weather for **{flight_num}**...")
    
    flight_data = get_specific_flight(flight_num)
    
    # Smart Regional Fallback based on Airline Prefix
    # Default to Vietnam (Hanoi/Ho Chi Minh region) for VJ (VietJet) or VN (Vietnam Airlines)
    if flight_num.startswith("VJ") or flight_num.startswith("VN"):
        lat, lon = 21.2212, 105.8072  # Hanoi (HAN) coordinates
    else:
        lat, lon = 51.4700, -0.4543   # London Heathrow default

    # If active radar tracking IS found, extract exact aircraft coordinates
    if flight_data and 'airport' in flight_data:
        try:
            lat = flight_data['airport']['origin']['position']['latitude']
            lon = flight_data['airport']['origin']['position']['longitude']
        except KeyError:
            pass
            
    # Fetch real-time weather for the correct airport coordinates
    weather_data = get_airport_weather(lat, lon)
    
    try:
        report: FlightReport = analyze_interactive_query(flight_data, weather_data, flight_num)
        
        status_emoji = "🟢" if report.status == "ON_TIME" else ("🟡" if report.status == "UNKNOWN" else "🔴")
        
        reply = (
            f"✈️ **FLIGHT REPORT: {report.flight_id.upper()}**\n\n"
            f"{status_emoji} **Status:** {report.status}\n"
            f"📈 **Delay Likelihood:** {report.delay_probability_percent}%\n"
            f"⏱️ **Predicted Delay:** +{report.predicted_delay_minutes} mins\n\n"
            f"📅 **Schedule Summary:**\n{report.schedule_summary}\n\n"
            f"🌤️ **Regional Weather Assessment:**\n{report.weather_summary}\n\n"
            f"🔍 **Detailed Reason & Risk Analysis:**\n{report.detailed_reasons}"
        )
        bot.edit_message_text(reply, chat_id=message.chat.id, message_id=status_msg.message_id, parse_mode="Markdown")
    except Exception as e:
        bot.edit_message_text(f"❌ Error analyzing query: {e}", chat_id=message.chat.id, message_id=status_msg.message_id)

def send_push_notification(alert: AutomatedAlert):
    """Pushes automated background warnings to your personal Telegram Chat ID."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    msg = (
        f"⚠️ **AUTOMATED AVIATION ALERT: {alert.severity}**\n\n"
        f"✈️ **Flight:** {alert.flight_id}\n"
        f"🌩️ **Hazard:** {alert.risk_factor}\n"
        f"⏱️ **Predicted Delay:** +{alert.predicted_delay_minutes} mins\n"
        f"💡 **Action:** {alert.recommended_action}"
    )
    requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"})

# ---------------------------------------------------------
# 6. Execution Entry Point
# ---------------------------------------------------------
if __name__ == "__main__":
    print("🚀 Aviation AI System Active...")
    # Start Telegram Bot Polling loop for interactive user queries
    bot.infinity_polling()
