import os
import weaviate
import google.generativeai as genai
from google.generativeai import types
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings

from langchain_community.vectorstores import Weaviate



os.environ["GEMINI_API_KEY"] = "AIzaSyDEiQ_vDfjf2t6iklLDyNpXEhql-C26rHM"
os.environ["WEAVIATE_APIKEY"] = "q6hPVic6EY6NZmLFV2HZH3MABq3aYh5X7c7I"

WEAVIATE_URL = "https://jf61yhukrgszmxekvubvag.c0.europe-west3.gcp.weaviate.cloud"
WEAVIATE_API_KEY = os.getenv("WEAVIATE_APIKEY")


file_path = "transcripts.txt"  
txt_loader = TextLoader(file_path, encoding="utf-8")
data = txt_loader.load()

print(f'You have {len(data)} documents in your data')
print(f'There are {len(data[0].page_content)} characters in your document')

# **Metni Küçük Parçalara Ayırma**
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", ". ", " ", ""])
docs = text_splitter.split_documents(data)

for i, chunk in enumerate(docs):
    print(f"Chunk {i+1}: {len(chunk.page_content)} characters")

# **Dummy Embeddings**
class DummyEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[0.0] * 1024] * len(texts)

    def embed_query(self, text):
        return [0.0] * 1024

embeddings = DummyEmbeddings()

# **Weaviate Bağlantısı**
auth_config = weaviate.auth.AuthApiKey(api_key=WEAVIATE_API_KEY)
client = weaviate.Client(
    url=WEAVIATE_URL,
    auth_client_secret=auth_config
)

# **Weaviate Schema Oluşturma**
client.schema.delete_all()
schema = {
    "classes": [
        {
            "class": "Chatbot",
            "description": "Documents for chatbot",
            "vectorizer": "text2vec-weaviate",
            "moduleConfig": {
                "text2vec-weaviate": {
                    "poolingStrategy": "cls",
                    "vectorizeClassName": False
                }
            },
            "properties": [
                {
                    "dataType": ["text"],
                    "description": "The content of the paragraph",
                    "moduleConfig": {
                        "text2vec-weaviate": {
                            "skip": False,
                            "vectorizePropertyName": False
                        }
                    },
                    "name": "content",
                }
            ]
        }
    ]
}
client.schema.create(schema)

# **Weaviate VectorStore Tanımlama**
vectorstore = Weaviate(
    client=client,
    index_name="Chatbot",
    text_key="content",
    attributes=["source"],
    embedding=embeddings
)

# **Veriyi Weaviate'e Yükleme**
text_meta_pair = [(doc.page_content, doc.metadata) for doc in docs]
texts, meta = list(zip(*text_meta_pair))
vectorstore.add_texts(texts, meta)

# **Weaviate + Gemini Chatbot**
class WeaviateGeminiChatbot:
    def __init__(self, index_name="Chatbot", model="gemini-1.5-pro"):
        """Weaviate + Gemini API Chatbot"""
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        self.model = model
        self.client = weaviate.Client(url=WEAVIATE_URL, auth_client_secret=auth_config)
        self.index_name = index_name
        self.chat_history = []

    def retrieve_documents(self, query, k=3):
        """Weaviate'den Dokümanları Getir"""
        try:
            results = self.client.query.get(
                class_name=self.index_name,
                properties=["content"]
            ).with_limit(k).do()

            docs = [doc["content"] for doc in results["data"]["Get"][self.index_name]]
            return docs
        except Exception as e:
            print(f"Weaviate Error: {e}")
            return []

    def chat(self, user_input, k=3):
        """Gemini API ile Cevap Üret"""
        retrieved_docs = self.retrieve_documents(user_input, k)

        context = "\n\n".join(retrieved_docs) if retrieved_docs else ""

        self.chat_history.append({"role": "user", "parts": [{"text": user_input}]})

        prompt = f"Context:\n{context}\n\nUser: {user_input}\nBot:"
        generate_content_config = genai.types.GenerationConfig(
            temperature=0.7,
            max_output_tokens=8192,
            top_p=0.95,
            top_k=40,
            response_mime_type="text/plain"
        )

        print("\nGemini Thinking...\n")
        response_text = ""

        # Gemini API Çağrısı
        model = genai.GenerativeModel(self.model)
        response = model.generate_content(
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
            generation_config=generate_content_config,
            stream=True
        )

        for chunk in response:
            print(chunk.text, end="", flush=True)
            response_text += chunk.text

        print("\n")
        self.chat_history.append({"role": "model", "parts": [{"text": response_text}]})

        return response_text

# **Chatbot Çalıştır**
if __name__ == "__main__":
    chatbot = WeaviateGeminiChatbot(index_name="Chatbot", model="gemini-1.5-pro")

    while True:
        user_input = input("\n You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Chatbot session closed.")
            break
        chatbot.chat(user_input)
