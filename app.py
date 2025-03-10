# app.py
import streamlit as st
import os
import sys
from chatbot import WeaviateGeminiChatbot  # Ana kodunuzdan chatbot sınıfını import edin

# Streamlit Uygulama Arayüzü
def main():
    st.set_page_config(
        page_title="Weaviate & Gemini Chatbot",
        page_icon="🤖",
        layout="centered"
    )

    # Streamlit sayfa başlığı ve açıklaması
    st.title("📚 Doküman Tabanlı Chatbot")
    st.subheader("Weaviate ve Gemini API ile Geliştirilmiş Sohbet Asistanı")
    
    # API anahtarlarını ve model seçimini düzenleme seçeneği
    with st.sidebar:
        st.header("Ayarlar")
        gemini_api = st.text_input("Gemini API Key", value=os.environ.get("GEMINI_API_KEY", ""), type="password")
        weaviate_api = st.text_input("Weaviate API Key", value=os.environ.get("WEAVIATE_APIKEY", ""), type="password")
        model_name = st.selectbox("Model Seçimi", ["gemini-1.5-pro", "gemini-1.5-flash"], index=0)
        k_docs = st.slider("Getirilen Doküman Sayısı", min_value=1, max_value=10, value=3)
        
        if st.button("Ayarları Güncelle"):
            os.environ["GEMINI_API_KEY"] = gemini_api
            os.environ["WEAVIATE_APIKEY"] = weaviate_api
            st.success("Ayarlar güncellendi!")
    
    # Oturumu başlat
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = WeaviateGeminiChatbot(
            index_name="Chatbot", 
            model="gemini-1.5-pro"
        )
    
    # Mesaj geçmişi
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Önceki mesajları göster
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Kullanıcı girdisi
    if prompt := st.chat_input("Bir soru sorun..."):
        # Kullanıcı mesajını ekle
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Asistan cevabı
        with st.chat_message("assistant"):
            # Streamlit ile uyumlu olması için özel konteyner oluştur
            message_placeholder = st.empty()
            
            # Stream fonksiyonunu değiştirmeden kullanmak için
            # bir StringIO benzeri nesne oluşturup kullanıyoruz
            class StreamCapture:
                def __init__(self):
                    self.content = ""
                
                def write(self, text):
                    self.content += text
                    message_placeholder.markdown(self.content)
                    return len(text)
                
                def flush(self):
                    pass
            
            # Stream çıktısını yakalayacak nesne
            stream_capture = StreamCapture()
            
            # Orijinal stdout'u kaydet
            original_stdout = sys.stdout
            
            try:
                # Çıktıyı yakalama nesnesine yönlendir
                sys.stdout = stream_capture
                
                # Chatbot'u çağır (terminal çıktısı artık Streamlit'e gidecek)
                response = st.session_state.chatbot.chat(prompt, k=k_docs)
                
                # Yakalanan yanıtı son haliyle göster
                message_placeholder.markdown(response)
            finally:
                # Orijinal stdout'u geri yükle
                sys.stdout = original_stdout
        
        # Asistan mesajını kaydet
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
