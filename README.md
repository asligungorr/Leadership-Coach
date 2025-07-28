# 🚀 **Leadership Coach Chatbot**

## 📘 Project Overview
The **Leadership Coach Chatbot** is an advanced **Retrieval‑Augmented Generation (RAG)** system for leadership development. It couples **Weaviate** (vector search) with the **Google Gemini API** (large‑language model) to generate intelligent, context‑aware answers grounded in a custom knowledge base built from real‑world audio transcripts.

---

## 🧰 Tech Stack
| Layer | Technology | Purpose |
|-------|------------|---------|
| Core  | **Python** | Main development language |
| UI    | **Streamlit** | Interactive web interface |
| Storage | **Weaviate** | Vector database for semantic retrieval |
| LLM   | **Google Gemini API** | Response generation (Gemini 1.5 Pro / Flash) |
| Orchestration | **LangChain** | Ingestion, retrieval, and prompt chaining |
| Transcription | **Whisper AI** | High‑accuracy speech‑to‑text |

---

## ⚙️ How It Works

### 📥 1. Knowledge Base Construction
1. `document_loader.py` → converts audio to text via **Whisper AI** → saves to `transcripts.txt`.
   > **Note:** Audio was sourced from a curated [YouTube playlist on leadership coaching](https://www.youtube.com/playlist?list=PLHSXjBr0dFpm4tx2qzAgGN8Dv98d9lZtx), and transcribed automatically using Whisper AI.    
3. `chatbot.py` → creates vector embeddings from transcripts → indexes them in **Weaviate**.

### 💬 2. Chatbot Interaction
1. A user query arrives through the **Streamlit** app (`app.py`).  
2. Relevant transcript chunks are retrieved from **Weaviate**.  
3. Those chunks are injected as context into **Google Gemini**, which returns a precise, well‑grounded reply.

---

## 🎯 Key Features
- 🔍 **RAG‑based Retrieval** – grounded answers backed by primary documents.  
- 🎙️ **Audio‑to‑Text Pipeline** – fully automated with **Whisper AI**.  
- 💡 **Contextual Intelligence** – leverages real leadership conversations.  
- 💬 **Live Chat Interface** – responsive Streamlit UI with streaming tokens.  
- 🔄 **Model Flexibility** – switch between *Gemini 1.5 Pro* and *Flash* effortlessly.

---

## 🎥 Demo
**[▶ Watch the demo](video1448249942.mp4)** — a quick walkthrough illustrating end‑to‑end transcription, retrieval, and answer generation.

---

## ✅ Use Cases
- Executive & leadership coaching  
- AI‑driven training simulations  
- Conversational knowledge assistants

---

## 📌 Roadmap
- Multilingual transcription & Q/A  
- Role‑specific dialogue fine‑tuning (coach, HR, mentor)  
- Calendar/task integrations for proactive coaching tips  

---

## 🌐 Live Deployment

📍 The app is deployed on Streamlit Cloud.  
🔗 Deployment link is available upon request.



