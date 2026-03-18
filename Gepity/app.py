import streamlit as st

st.set_page_config(page_title="SmartDoc AI", layout="wide")

st.title("🤖 SmartDoc AI")
st.caption("Upload PDF và đặt câu hỏi")

# Sidebar
with st.sidebar:
    st.header("⚙️ Cài đặt")
    st.info("Model: Qwen2.5:7b")

# Main area
uploaded_file = st.file_uploader("📄 Upload PDF", type="pdf")

if uploaded_file:
    st.success(f"✅ Đã upload: {uploaded_file.name}")

question = st.text_input("💬 Đặt câu hỏi về tài liệu...")