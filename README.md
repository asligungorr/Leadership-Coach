#  ****Leadership Coach Chatbot****

 **Project Overview**

This project is an advanced Retrieval-Augmented Generation (RAG) Chatbot powered by Weaviate as a vector database and Google Gemini API as the generative model. The chatbot leverages a knowledge base extracted from transcripts, enabling intelligent and context-aware responses based on real-world documents.

---

 **Tech Stack**

- **Python**: Core programming language  
- **Streamlit**: Interactive UI for the chatbot  
- **Weaviate**: Vector database for storing and retrieving document embeddings  
- **Google Gemini API**: Large language model for generating responses  
- **LangChain**: Managing document ingestion and retrieval  
- **Whisper AI**: Automated speech-to-text transcription for building the knowledge base  

---

 **How It Works**

###  **Knowledge Base Preparation**
- The **document_loader.py** script extracts text from audio files using **Whisper AI** and saves it to `transcripts.txt`.  
- The **chatbot.py** script processes these transcripts and stores them as **vector embeddings** in **Weaviate**.  

###  **Chatbot Interaction**
- User queries are processed in **app.py**, where the chatbot retrieves the most relevant document chunks from **Weaviate**.  
- Retrieved documents are provided as context to **Google Gemini**, generating an **enhanced response**.  

---

 **Key Features**

-  **RAG-based Retrieval** - Uses Weaviate to fetch relevant knowledge chunks before generating responses.  
-  **Streaming Chat Interface** - Built with Streamlit for an interactive experience.  
-  **Audio-to-Text Pipeline** - Uses Whisper AI to generate text-based knowledge from audio sources.  
-  **Multi-Model Support** - Easily switch between Gemini-1.5-Pro and Gemini-1.5-Flash models.

##  ****Demo Video**** 

**[Watch the demo]** video1448249942.mp4












