import streamlit as st
import os
import sys
from chatbot import WeaviateGeminiChatbot  

# Streamlit Application Interface
def main():
    st.set_page_config(
        page_title="Leadership Coach Chatbot",
        page_icon="🤖",
        layout="centered"
    )

    
    st.title("📚 Document-Based Chatbot")
    st.subheader("Developed Chat Assistant with Weaviate and Gemini API")
    
    # Option to Edit API Keys and Model Selection
    with st.sidebar:
        st.header("Settings")
        gemini_api = st.text_input("Gemini API Key", value=os.environ.get("GEMINI_API_KEY", ""), type="password")
        weaviate_api = st.text_input("Weaviate API Key", value=os.environ.get("WEAVIATE_APIKEY", ""), type="password")
        model_name = st.selectbox("Select Model", ["gemini-1.5-pro", "gemini-1.5-flash"], index=0)
        
        if st.button("Update Settings"):
            os.environ["GEMINI_API_KEY"] = gemini_api
            os.environ["WEAVIATE_APIKEY"] = weaviate_api
            st.success("The settings have been updated!")
    
    # Start Session
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = WeaviateGeminiChatbot(
            index_name="Chatbot", 
            model="gemini-1.5-pro"
        )
    
    # Message History
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Show Previous Messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # User Input
    if prompt := st.chat_input("Ask a question..."):
        # Add User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Assistant Response
        with st.chat_message("assistant"):
            # Create a Custom Container for Streamlit Compatibility
            message_placeholder = st.empty()
            
            # To use the stream function without modification, we create and use a StringIO-like object.
            class StreamCapture:
                def __init__(self):
                    self.content = ""
                
                def write(self, text):
                    self.content += text
                    message_placeholder.markdown(self.content)
                    return len(text)
                
                def flush(self):
                    pass
            
            # An object to capture stream output
            stream_capture = StreamCapture()
            
            # Save the original stdout
            original_stdout = sys.stdout
            
            try:
                # Redirect output to the capture object
                sys.stdout = stream_capture
                
                # Call the Chatbot (terminal output will now be redirected to Streamlit)
                response = st.session_state.chatbot.chat(prompt, k=3)
                
                # Display the Captured Response in its Final Form
                message_placeholder.markdown(response)
            finally:
                # Restore the original stdout
                sys.stdout = original_stdout
        
        # Save Assistant Message
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
