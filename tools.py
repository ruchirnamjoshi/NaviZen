from langchain_community.document_loaders import JSONLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate,ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.docstore.document import Document
from langchain.agents import Tool, initialize_agent, create_openai_functions_agent, AgentExecutor
from langchain.agents.agent import RunnableAgent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.chat_models import ChatOpenAI
from langchain_chroma import Chroma
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Optional
import pandas as pd
import os
import json
import gradio as gr
from gtts import gTTS
import tempfile
import base64
import numpy as np
import requests
import pandas as pd
from tqdm import tqdm
import csv
from datetime import datetime
import os

load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ORS_API_KEY = os.getenv("ORS_API_KEY")

# === SETUP ===
DATASET_PATH = "AutoRAG_Dataset"
manual_path = os.path.join(DATASET_PATH, "car_manuals", "manuals.json")
weather_path = os.path.join(DATASET_PATH, "weather_data", "weather_data.csv")
poi_path = os.path.join(DATASET_PATH, "poi_data", "poi_data.csv")
PROFILE_PATH = os.path.join(DATASET_PATH, "driver_profiles", "profiles.json")
TELEMETRY_PATH = os.path.join(DATASET_PATH, "telemetry", "telemetry_logs.csv")
VOICE_LOG_PATH = os.path.join(DATASET_PATH, "voice_queries", "voice_queries.csv")
CALENDAR_PATH = os.path.join(DATASET_PATH, "calendar_events", "calendar_events.csv")

# === DATA LOADING FUNCTIONS ===
def load_weather_data():
    return pd.read_csv(weather_path)

def load_poi_data():
    return pd.read_csv(poi_path)

# === Weather Tool ===
class WeatherInput(BaseModel):
    city: str

@tool("get_weather", args_schema=WeatherInput)
def get_weather_tool(city: str) -> str:
    """Get current weather info for a city using Open-Meteo API."""
    url = f"https://api.open-meteo.com/v1/forecast?latitude=40.71&longitude=-74.01&current_weather=true"
    response = requests.get(url)
    if response.ok:
        data = response.json()
        weather = data['current_weather']
        return f"Current weather in {city}: {weather['temperature']}°C, Wind {weather['windspeed']} km/h."
    return "Sorry, unable to fetch weather."


# === Navigation Tool ===
class NavigationInput(BaseModel):
    from_location: str
    to_location: str
    

@tool("get_directions", args_schema=NavigationInput)
def get_directions_tool(from_location: str, to_location: str) -> str:
    """Get driving directions from one location to another using OpenRouteService."""
    def geocode(location: str) -> list:
        geo_url = f"https://api.openrouteservice.org/geocode/search"
        headers = {
            "Authorization": ORS_API_KEY
        }
        params = {
            "api_key": ORS_API_KEY,
            "text": location,
            "size": 1
        }
        resp = requests.get(geo_url, headers=headers, params=params)
        try:
            return resp.json()['features'][0]['geometry']['coordinates']
        except Exception as e:
            raise ValueError(f"Could not geocode '{location}': {resp.text}")

    headers = {
        "Authorization": ORS_API_KEY,
        "Content-Type": "application/json"
    }

    try:
        from_coords = geocode(from_location)
        to_coords = geocode(to_location)
    except ValueError as e:
        return str(e)

    body = {
        "coordinates": [from_coords, to_coords]
    }

    url = "https://api.openrouteservice.org/v2/directions/driving-car"
    response = requests.post(url, headers=headers, json=body)

    try:
        data = response.json()
    except:
        return f"Invalid JSON returned: {response.text}"

    if "features" not in data:
        return f"Unexpected response format:\n{data}"

    try:
        steps = data['features'][0]['properties']['segments'][0]['steps']
        directions = "\n".join([f"{i+1}. {s['instruction']}" for i, s in enumerate(steps)])
        return "Navigation Steps:\n" + directions
    except Exception as e:
        return f"Failed to parse directions: {e}\nFull response:\n{data}"


# === User Profile Tool ===
class UserProfileInput(BaseModel):
    name: str

@tool("get_user_profile", args_schema=UserProfileInput)
def get_user_profile_tool(name: str) -> str:
    """Retrieve driver profile based on name."""
    with open(PROFILE_PATH) as f:
        profiles = json.load(f)

    name = name.strip().lower()

    for p in profiles:
        if p['name'].split()[0].lower() == name:
            return (
                f"User {p['name']}: {p['age']} yrs\n"
                f"Home: {p['home_address']}\n"
                f"Work: {p['work_address']}\n"
                f"Driving Style: {p['driving_style']}\n"
                f"Preferred Music: {p['preferred_music']}\n"
                f"Saved Locations:\n- " + "\n- ".join(p['saved_locations'])
            )
    
    return "User not found."


# === RAG Retrieval Tool ===
class RAGInput(BaseModel):
    query: str

@tool("smart_rag", args_schema=RAGInput)
def smart_rag_tool(query: str) -> str:
    """Answer queries using in-vehicle data including manuals, FAQs, and Points Of Interests, trip logs,Telemetry Logs,Voice Query History,Calander Events  ."""
    vectorstore = FAISS.load_local("car_manual_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever()
    docs = retriever.get_relevant_documents(query)
    return "\n".join([doc.page_content for doc in docs]) if docs else "No relevant info found in the manual."




class LogFullInteractionInput(BaseModel):
    name: str         # Full name input
    query: str        # User's spoken/typed input
    response: str     # LLM model's response

@tool("log_user_interaction", args_schema=LogFullInteractionInput)
def log_user_interaction_tool(name: str, query: str, response: str) -> str:
    """Logs a user query and the corresponding model response."""
    # Load existing profiles to resolve name to user_id
    with open(PROFILE_PATH) as f:
        profiles = json.load(f)

    name = name.strip().lower()
    matched_user = next((p for p in profiles if name in p["name"].lower()), None)
    if not matched_user:
        return f"⚠️ No user found matching name '{name}'"

    user_id = matched_user["user_id"]
    full_name = matched_user["name"]
    timestamp = datetime.utcnow().isoformat()
    intent = query.strip().split()[0].lower() if query.strip() else "unknown"

    log_entry = {
        "user_id": user_id,
        "name": full_name,
        "timestamp": timestamp,
        "query_text": query,
        "response_text": response,
        "intent": intent
    }

    # Ensure file has header if it doesn't exist
    file_exists = os.path.exists(VOICE_LOG_PATH)
    with open(VOICE_LOG_PATH, "a", newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["user_id", "name", "timestamp", "query_text", "response_text", "intent"]
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_entry)

    return f"✅ Logged interaction for {full_name} ({user_id})"