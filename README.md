# 🚗 NaviZen: The Intelligent In-Vehicle Voice Co-Pilot

NaviZen is a powerful, intelligent, voice-first assistant for vehicles that blends Retrieval-Augmented Generation (RAG), real-time APIs, and personalized driver data to deliver **context-aware, natural conversations on the road**.

> 🧠 Built with LangChain · FAISS · OpenAI GPT-4o · Gradio · Python · OpenRouteService · Open-Meteo

---

## 🌟 Key Features

### 🔍 Retrieval-Augmented Generation (RAG)
- Combines structured vehicle data, user history, manuals, trip telemetry, and POIs.
- Powered by vector search with FAISS for fast, relevant context.

### 🗣️ Voice + Text Input
- Supports both **spoken and typed queries** via Gradio UI.
- Converts responses into **natural-sounding speech** using `gTTS`.

### 📕 Car Manual Q&A
- Ask anything from car features to maintenance tips.
- Answers are retrieved from indexed car manuals.

### 👤 Personalized Driver Profiles
- Retrieves driver-specific data such as:
  - Name, age, preferred music, saved locations
  - Home/work address and driving style

### 🧾 Trip Telemetry Summaries
- Pulls historical driving data:
  - Speed logs, tire pressure, cabin temperature, battery levels

### 📍 Real-Time POI & Navigation
- Integrates with OpenRouteService API
- Provides step-by-step **driving directions**
- Lists nearby **restaurants, charging stations, hospitals**, etc.

### ☁️ Live Weather Info
- Uses Open-Meteo API to provide **current weather updates** by city.

### 📅 Calendar Integration
- Lists your **upcoming appointments** from car calendar memory.

### 🔁 Conversation Logging
- Every user interaction is **automatically logged**:
  - Voice query
  - AI-generated response
  - Inferred intent
  - Timestamp
- Logs are saved in a persistent CSV file for future retrieval.

---

## 🧰 Tech Stack

| Tool/Library        | Purpose                             |
|---------------------|-------------------------------------|
| `LangChain`         | Tool abstraction, memory, agent logic |
| `FAISS`             | Fast vector similarity search       |
| `OpenAI GPT-4o`     | LLM for all natural language tasks  |
| `Gradio`            | Voice + UI interface                |
| `gTTS`              | Text-to-speech synthesis            |
| `OpenRouteService`  | Geocoding & driving directions      |
| `Open-Meteo`        | Real-time weather info              |
| `Pandas / JSON`     | Data parsing and synthetic generation |

---

## 📂 Data Types Indexed in Vector Store

- ✅ Car manuals (maintenance, parts, configuration)
- ✅ QA Triplets (FAQs + pre-written answers)
- ✅ POIs (from OpenRouteService geodata)
- ✅ Telemetry logs (trip data from 50 drivers)
- ✅ Calendar events
- ✅ Driver profiles
- ✅ Voice query logs (updated live)

---

## 🧠 How It Works

1. **Index Building**:
    - Parses car manuals, user logs, telemetry, etc.
    - Chunks and embeds all data using `OpenAIEmbeddings`.
    - Stores in FAISS index for retrieval.

2. **Agent Execution**:
    - Uses `RunnableAgent` + `ChatPromptTemplate` + tools
    - Prompt is injected with vector context + memory

3. **Tools**:
    - `smart_rag_tool` – RAG-powered response engine
    - `get_weather`, `get_directions` – external API fetchers
    - `get_user_profile`, `log_user_interaction` – local JSON/CSV tools

4. **User Interaction**:
    - Query → Agent response → Voice + Text return
    - Response is spoken aloud and also returned in text

---

## 💡 Why NaviZen?

- ✔️ Voice-friendly and road-safe
- ✔️ Context-aware, like a real human co-pilot
- ✔️ Extensible architecture — add new tools easily
- ✔️ Ideal for smart infotainment or autonomous vehicle platforms

---

## 🚀 Try It Yourself

```bash
git clone https://github.com/your-username/NaviZen.git
cd NaviZen
pip install -r requirements.txt
python app.py
```

Set your environment variables:
```env
OPENAI_API_KEY=sk-proj-...
ORS_API_KEY=your_openrouteservice_key
```

---

## 📣 About the Creator

Built with ❤️ by **Ruchir Namjoshi**, CS Grad Student @ Georgia State University.  
📬 [LinkedIn](https://www.linkedin.com/in/ruchir-namjoshi-687b86192/)

Looking for opportunities in **AI/ML, LLM Engineering, and Intelligent Systems**.

---

## 🏁 Future Additions

- 🔄 Continual learning from new queries
- 🧠 Driver mood & sentiment detection
- 🎵 Music playback & media integration
- 🚘 Integration into actual car OS dashboards

---
