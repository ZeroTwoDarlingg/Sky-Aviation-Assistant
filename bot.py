import os
import json
import requests
import telebot
from pydantic import BaseModel, Field
from google import genai
from FlightRadarAPI import FlightRadar24API

# 1. Load Secret Keys from Environment Variables
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

# Initialize Clients
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)
client = genai.Client(api_key=GEMINI_API_KEY)
fr_api = FlightRadar24API()

# 2. Define AI Response Schema
class FlightReport(BaseModel):
    flight_id: str = Field(description="Flight callsign or number")
    status: str = Field(description="ON_TIME, DELAYED, CANCELLED, or UNKNOWN")
    delay_probability_percent: int = Field(description="0 to 100 percentage likelihood of delay")
    predicted_delay_minutes: int = Field(description="Estimated delay duration in minutes")
    schedule_summary: str = Field(description="Departure, Arrival, and Time summary")
    weather_summary: str = Field(description="Summary of weather at origin or destination")
    detailed_reasons: str = Field(description="In-depth explanation of delays or risks (e.g., thunderstorms, wind shear, ATC congestion, drone/unidentified aircraft holds, technical maintenance)")

# 3. Data Retrieval Helpers
def get_flight_info(flight_number: str):
    """Searches live FlightRadar24 telemetry for the given flight number."""
    try:
        clean_no = flight_number.upper().strip().replace(" ", "")
        flights = fr_api.get_flights(flight_name=clean_no)
        if not flights:
            return None
        return fr_api.get_flight_details(flights[0])
    except Exception as e:
        print(f"Error fetching flight radar: {e}")
        return None

def get_airport_weather(lat: float, lon: float):
    """Fetches real-time weather from Open-Meteo (No key required)."""
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,wind_speed_10m,weather_code,precipitation&hourly=visibility,wind_gusts_10m"
        res = requests.get(url, timeout=10)
        return res.json().get("current", {})
    except Exception:
        return {}

def analyze_with_gemini(flight_data, weather_data, flight_query: str) -> FlightReport:
    """Uses Gemini 2.5 Flash to predict delays and identify root causes."""
    prompt = f"""
    You are an expert aviation intelligence AI.
    A user asked for monitoring on Flight Number: {flight_query}
    
    Live Flight Data: {json.dumps(flight_data if flight_data else "No active radar track found. Flight may be scheduled later today or grounded.")}
    Live Weather Data: {json.dumps(weather_data)}
    
    Tasks:
    1. Extract schedule details (Origin, Destination, Times).
    2. Analyze weather conditions (winds, storms, precipitation, visibility).
    3. Predict delay probability (0-100%) and estimated delay duration in minutes.
    4. Provide detailed reasoning for delays or hazards. Consider:
       - Meteorological factors (thunderstorms, fog, wind shear).
       - Airspace restrictions (ATC holds, drone sightings, security/airspace warnings).
       - Operational factors (late incoming aircraft, maintenance).
    """
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config={
            'response_mime_type': 'application/json',
            'response_schema': FlightReport,
        },
    )
    return response.parsed

# 4. Telegram Message Handlers
@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    welcome_msg = (
        "✈️ **AI Flight & Weather Tracker Bot**\n\n"
        "Send me any **Flight Number** (e.g., `VN123`, `AA100`, `DL456`).\n\n"
        "I will fetch real-time radar tracking, airport weather, calculate delay risk, "
        "and explain any disruption causes!"
    )
    bot.reply_to(message, welcome_msg, parse_mode="Markdown")

@bot.message_handler(func=lambda message: True)
def handle_flight_query(message):
    flight_num = message.text.strip().upper()
    status_msg = bot.reply_to(message, f"🔍 Fetching radar track and weather for **{flight_num}**...")
    
    # 1. Fetch Radar & Weather Data
    flight_data = get_flight_info(flight_num)
    
    lat, lon = 51.4700, -0.4543  # Default coordinates (London Heathrow) if unlisted
    if flight_data and 'airport' in flight_data:
        try:
            lat = flight_data['airport']['origin']['position']['latitude']
            lon = flight_data['airport']['origin']['position']['longitude']
        except KeyError:
            pass
            
    weather_data = get_airport_weather(lat, lon)
    
    # 2. Run AI Analysis
    try:
        report: FlightReport = analyze_with_gemini(flight_data, weather_data, flight_num)
        
        status_emoji = "🟢" if report.status == "ON_TIME" else "🔴"
        
        response_text = (
            f"✈️ **FLIGHT REPORT: {report.flight_id}**\n\n"
            f"{status_emoji} **Status:** {report.status}\n"
            f"📈 **Delay Likelihood:** {report.delay_probability_percent}%\n"
            f"⏱️ **Predicted Delay:** +{report.predicted_delay_minutes} mins\n\n"
            f"📅 **Schedule Summary:**\n{report.schedule_summary}\n\n"
            f"🌤️ **Weather Assessment:**\n{report.weather_summary}\n\n"
            f"🔍 **Detailed Disruption Reason:**\n{report.detailed_reasons}"
        )
        
        bot.edit_message_text(response_text, chat_id=message.chat.id, message_id=status_msg.message_id, parse_mode="Markdown")
    except Exception as e:
        bot.edit_message_text(f"❌ Error analyzing flight: {e}", chat_id=message.chat.id, message_id=status_msg.message_id)

# 5. Start Polling Loop
if __name__ == "__main__":
    print("Bot initialized! Listening 24/7 for Telegram queries...")
    bot.infinity_polling()