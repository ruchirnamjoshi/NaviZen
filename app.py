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
from build_vectorstore import build_vectorstore_from_multiple_sources
from tools import get_weather_tool, get_directions_tool, get_user_profile_tool, smart_rag_tool, log_user_interaction_tool
import os



load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ORS_API_KEY = os.getenv("ORS_API_KEY")

# Check the key

if not OPENAI_API_KEY:
    print("No OpenAI API key was found - please head over to the troubleshooting notebook in this folder to identify & fix!")
elif not OPENAI_API_KEY.startswith("sk-proj-"):
    print("An OpenAI API key was found, but it doesn't start sk-proj-; please check you're using the right key - see troubleshooting notebook")
elif OPENAI_API_KEY.strip() != OPENAI_API_KEY:
    print("An OpenAI API key was found, but it looks like it might have space or tab characters at the start or end - please remove them - see troubleshooting notebook")
else:
    print("OpenAi API key found and looks good so far!")
    
if not ORS_API_KEY:
    print("No ORS API key was found - please head over to the troubleshooting notebook in this folder to identify & fix!")
elif ORS_API_KEY.strip() != ORS_API_KEY:
    print("An API key was found, but it looks like it might have space or tab characters at the start or end - please remove them - see troubleshooting notebook")
else:
    print("ORS API key found and looks good so far!")


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

def get_agent_executor():
    tools = [
        get_weather_tool,
        get_directions_tool,
        get_user_profile_tool,
        smart_rag_tool,
        log_user_interaction_tool
    ]

    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
            """
            You are AutoRAG++, an intelligent in-vehicle assistant for drivers.

            You have access to various tools and a rich internal knowledge base powered by vector search. 
            The knowledge base contains multiple types of documents and data that you can retrieve and use to answer queries accurately and helpfully.
            
            Here’s what your memory includes:
            
            📘 **Car Manuals**: Detailed information about car parts, maintenance instructions, settings, and troubleshooting steps.
            
            💬 **QA Triplets**: Frequently asked questions from users with context and pre-generated answers.
            
            📍 **Points of Interest (POIs)**: Places like restaurants, hospitals, and gas stations along with their type, location, rating, and distance.
            
            👤 **Driver Profiles**: Data about each user including name, age, home and work addresses, driving style, preferred music, and saved locations.
            
            📊 **Telemetry Logs**: Trip-level data including speed, battery level, tire pressure, and cabin temperature over time for each user.
            
            🗣️ **Voice Query History**: Past voice/text interactions made by users with associated timestamps and inferred intent.
            
            📅 **Calendar Events**: Upcoming user events such as meetings or service appointments with location and time.
            
    
            
            🛠️ You also have access to tools that allow you to:
            - Navigate between places using real-time routing
            - Retrieve current weather
            - Look up specific user profiles
            - Log new user interactions (query + response)
            - Retrieve relevant information using RAG (retrieval-augmented generation)
            
            🎯 Your job is to use this knowledge and the tools to help the driver:
            - Answer questions about car functionality or past trips
            - Find POIs or weather in any location
            - Understand driving stats or calendar reminders
            - Act as a friendly voice assistant that logs every interaction
            - Keep responses short, clear, and spoken in a natural way

            You should log every user query and your response using the `log_user_interaction` tool. Provide the user's name, query, and your response.
            Always try to personalize your responses using user-specific data and keep answers natural for speech (as your response will be read aloud). Do not use markdown or any formating.
            """
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    prompt = prompt.partial()

    agent: RunnableAgent = create_openai_functions_agent(
        llm=llm,
        prompt=prompt,
        tools=tools
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True
    )

    return executor

# === UI ===
def launch_ui(chain):
    def respond(audio_file, text):
        if text:
            query = text
        elif audio_file:
            import speech_recognition as sr
            recognizer = sr.Recognizer()
            with sr.AudioFile(audio_file) as source:
                audio = recognizer.record(source)
            try:
                query = recognizer.recognize_google(audio)
            except sr.UnknownValueError:
                return gr.update(value=""), "Sorry, I couldn't understand the audio.", ""
            except sr.RequestError:
                return gr.update(value=""), "Speech recognition service error.", ""
        else:
            return gr.update(value=""), "Please provide input.", ""

        result = chain.invoke({"input": query})["output"]
        

        tts = gTTS(result)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            temp_audio_path = f.name
            tts.save(temp_audio_path)

        with open(temp_audio_path, "rb") as f:
            audio_data = f.read()
        audio_base64 = base64.b64encode(audio_data).decode("utf-8")

        audio_html = f"""
        <audio id='response-audio' controls autoplay style='width:100%;'>
            <source src='data:audio/mpeg;base64,{audio_base64}' type='audio/mpeg'>
            Your browser does not support the audio element.
        </audio>
        <script>
            const audio = document.getElementById("response-audio");
            if (audio) {{
                audio.playbackRate = 1.5;
                audio.play().catch(() => {{ console.warn("Autoplay blocked."); }});
            }}
        </script>
        """
        return gr.update(value=""), result, audio_html
    
    gr.Interface(
        fn=respond,
        inputs=[
            gr.Audio(type="filepath", label="🎙️ Speak your query (or type below)"),
            gr.Textbox(label="✏️ Or type your query")
        ],
        outputs=[
            gr.Textbox(label="✏️", interactive=True),
            gr.Textbox(label="🧠 Response", interactive=False),
            gr.HTML(label="🔊 Audio Reply")
        ],
        
        title="🚗 NaviZen — A calm, reliable In-Vehicle voice co-pilot that understands you.",
        theme="soft"
    ).launch()

# === MAIN RUN ===
if __name__ == "__main__":
    print("🚀 Building index and launching assistant...")
    build_vectorstore_from_multiple_sources(index_path="car_manual_index")
    weather_df = load_weather_data()
    poi_df = load_poi_data()
    rag_chain = get_agent_executor()
    launch_ui(rag_chain)