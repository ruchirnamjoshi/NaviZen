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

# === VECTOR INDEX BUILDING FROM MULTIPLE SOURCES ===
def build_vectorstore_from_multiple_sources(index_path="car_manual_index"):
    documents = []

    
    # Car Manuals
    with open(manual_path) as f:
        manuals = json.load(f)
    for m in manuals:
        documents.append(Document(
            page_content=m["content"],
            metadata={"type": "manual", "doc_id": m["doc_id"]}
        ))


    # QA Triplets
    with open("AutoRAG_Dataset/qa_triplets/qa_pairs.json") as f:
        qas = json.load(f)
    for qa in qas:
        content = f"Q: {qa['question']}\nContext: {qa['retrieved_context']}\nA: {qa['answer']}"
        documents.append(Document(
            page_content=content,
            metadata={"type": "qa_triplet", "qa_id": qa["qa_id"]}
        ))


    # POI
    poi_df = load_poi_data()
    for _, row in poi_df.iterrows():
        content = f"{row['type']} - {row['name']} in {row['city']}, Rating: {row['rating']}, Distance: {row['distance_km']} km"
        documents.append(Document(
            page_content=content,
            metadata={"type": "poi", "city": row["city"], "poi_type": row["type"]}
        ))

    # === Helper ===
    def format_saved_locations(locs):
        return "; ".join(locs)
    
    # === 1. Driver Profiles ===
    with open(PROFILE_PATH) as f:
        profiles = json.load(f)
    
    profile_lookup = {p["user_id"]: p for p in profiles}
    
    for profile in profiles:
        content = (
            f"This is information about user {profile['name']} (user ID: {profile['user_id']}). "
            f"They are {profile['age']} years old and prefer {profile['preferred_music']} music. "
            f"Their home address is {profile['home_address'].replace(chr(10), ', ')}, and they work at {profile['work_address'].replace(chr(10), ', ')}. "
            f"Their driving style is {profile['driving_style']}. "
            f"Saved locations include: {format_saved_locations(profile['saved_locations'])}."
        )
        documents.append(Document(page_content=content, metadata={"type": "profile", "user_id": profile['user_id']}))
    
    # === 2. Telemetry Logs ===
    telemetry_df = pd.read_csv(TELEMETRY_PATH)
    for user_id, user_df in tqdm(telemetry_df.groupby("user_id"), desc="Processing telemetry"):
        profile = profile_lookup.get(user_id, {})
        user_name = profile.get("name", user_id)
        summary = (
            f"Telemetry summary for user {user_name} (user ID: {user_id}). "
            f"{len(user_df)} data points recorded across trips. "
            f"Speed ranged from {user_df['speed_kmph'].min()} to {user_df['speed_kmph'].max()} km/h. "
            f"Battery ranged from {user_df['battery_level_percent'].min()}% to {user_df['battery_level_percent'].max()}%. "
            f"Tire pressure was typically around {user_df['tire_pressure_psi'].mean():.2f} PSI. "
            f"Cabin temperature averaged {user_df['cabin_temp_c'].mean():.2f}°C."
        )
        documents.append(Document(page_content=summary, metadata={"type": "telemetry", "user_id": user_id}))

    # === 3. Voice Queries ===
    queries_df = pd.read_csv(VOICE_LOG_PATH)
    for user_id, user_df in tqdm(queries_df.groupby("user_id"), desc="Processing voice queries"):
        profile = profile_lookup.get(user_id, {})
        user_name = profile.get("name", user_id)
        content = (
            f"Voice interaction history for user {user_name} (user ID: {user_id}): "
            f"They asked the assistant the following types of queries: "
            f"{'; '.join(user_df['query_text'].unique()[:10])}."
        )
        documents.append(Document(page_content=content, metadata={"type": "voice_queries", "user_id": user_id}))
    
    # === 4. Calendar Events ===
    calendar_df = pd.read_csv(CALENDAR_PATH)
    for user_id, user_df in tqdm(calendar_df.groupby("user_id"), desc="Processing calendar"):
        profile = profile_lookup.get(user_id, {})
        user_name = profile.get("name", user_id)
        events = ", ".join(user_df["title"].unique())
        content = (
            f"Calendar for user {user_name} (user ID: {user_id}): "
            f"Scheduled events include: {events}. "
            f"Typical durations are around {user_df['duration_minutes'].mean():.1f} minutes."
        )
        documents.append(Document(page_content=content, metadata={"type": "calendar", "user_id": user_id}))


    # Embedding
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(split_docs, embeddings)
    db.save_local(index_path)
    return db

