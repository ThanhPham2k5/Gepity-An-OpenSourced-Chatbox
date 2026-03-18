import streamlit as st
from langchain_ollama import OllamaLLM

# setup
st.set_page_config(page_title="Gepity AI", layout="wide")

llm = OllamaLLM(
    model="qwen2.5:7b",
    base_url="http://172.25.64.1:11434"
)

# get request from user
user_req = st.chat_input(placeholder="Nhập yêu cầu của bạn...")

if "messages" not in st.session_state:
    st.session_state.messages = []

# store response into session
if user_req != None:
    response = llm.invoke(user_req);
    st.session_state.messages.append({"role": "user", "content": user_req})
    st.session_state.messages.append({"role": "ai", "content": response})

# response (entire history)
for messages in st.session_state.messages:
    with st.chat_message(messages["role"]):
        st.write(messages["content"])
