import streamlit as st
import os
from datetime import datetime 
import streamlit.components.v1 as components
from utils.helpers import img_to_base64, update_current_chat_to_history
from core import RAG_engine


# SETUP & SESSIONS -------------------------------------------------------
st.set_page_config(page_title="Gepity AI", layout="wide")

if "rag" not in st.session_state:
    st.session_state.rag = RAG_engine()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None

# LOAD CSS -------------------------------------------------------
# Get the directory where app.py actually lives
current_dir = os.path.dirname(os.path.abspath(__file__))
# Build the full path to demo.css
css_path = os.path.join(current_dir, "demo.css")
def load_css(file_path):
    with open(file_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css(css_path)

# PREPARE ASSETS -------------------------------------------------------
trash_can = img_to_base64("assets/img/trash-can.png")
section_ico = img_to_base64("assets/img/section-ico.png")
user_bubble = img_to_base64("assets/img/user-ico.png")
ai_bubble = img_to_base64("assets/img/ai-ico.png")

# SIDEBAR SECTION -------------------------------------------------------
with st.sidebar:
    # Header & Logo
    st.markdown("""
        <div class="logo-container">
            <div class="logo-box">G</div>
            <div class="logo-text">Gepity</div>
        </div>
        <div class="logo-sub">AI Document Q&A System</div>
    """, unsafe_allow_html=True)
    
    # Action Button
    if st.button("+ Cuộc trò chuyện mới", type="primary"):
        update_current_chat_to_history()
        
        st.session_state.messages = []
        st.session_state.current_chat_id = None
        st.session_state.retriever = None
        st.session_state.vector_store = None
        if "last_file_key" in st.session_state:
            del st.session_state["last_file_key"]
        if "file_stats" in st.session_state:
            del st.session_state["file_stats"]
        st.rerun()

    st.markdown('<div class="sections-title">LỊCH SỬ TRÒ CHUYỆN</div>', unsafe_allow_html=True)
    with st.container(border=False):
        for chat in st.session_state.chat_history:
            if st.button(chat['title'], key=f"hist_{chat['id']}", use_container_width=True, type="secondary"):
                update_current_chat_to_history()
                
                st.session_state.messages = chat['messages'].copy()
                st.session_state.current_chat_id = chat['id']
                st.session_state.retriever = chat.get('retriever')
                st.session_state.vector_store = chat.get('vector_store')
                st.session_state['last_file_key'] = chat.get('last_file_key')
                if "file_stats" in st.session_state:
                    del st.session_state["file_stats"]

                st.rerun()

    # Section: Hướng dẫn
    st.markdown('<div class="sidebar-label">Hướng dẫn sử dụng</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="step-item">
            <div class="step-num">1</div>
            <div>Upload file PDF hoặc DOCX cần hỏi</div>
        </div>
        <div class="step-item">
            <div class="step-num">2</div>
            <div>Chờ hệ thống xử lý tài liệu</div>
        </div>
        <div class="step-item">
            <div class="step-num">3</div>
            <div>Nhập câu hỏi về nội dung tài liệu</div>
        </div>
        <div class="step-item">
            <div class="step-num">4</div>
            <div>Xem câu trả lời và tiếp tục hỏi</div>
        </div>
    """, unsafe_allow_html=True)

    # Section: Cấu hình Model
    st.markdown('<div class="sidebar-label">Cấu hình hệ thống</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="config-card">
            <div class="config-key">Mô hình đang dùng</div>
            <div class="config-val">qwen2.5:7b</div>
            <div class="status-online">
                <div class="dot"></div> Online
            </div>
        </div>
        <div class="config-card">
            <div class="config-key">Embedding Model</div>
            <div class="config-val">multilingual-mpnet</div>
        </div>
        <div class="config-card">
            <div class="config-key">Vector Database</div>
            <div class="config-val">FAISS</div>
        </div>
    """, unsafe_allow_html=True)

# HEADER SECTION -------------------------------------------------------
st.markdown(f"""
    <div class="main-header">
        <div class="header-left">
            <div class="header-left-main">SmartDoc AI — Gepity</div>
            <div class="header-left-sub">Hỏi đáp thông minh từ tài liệu của bạn</div>
        </div>
        <div class="header-right">
            <div class="header-status-dot"></div>
            <div class="header-status-text">qwen2.5:7b</div>
        </div>
    </div>
""", unsafe_allow_html=True)


# CHAT SECTION -------------------------------------------------------
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        is_user = msg["role"] == "user"
        bubble_class = "user-bubble" if is_user else "ai-bubble"
        name = "Bạn" if is_user else "Gepity"
        
        st.markdown(f"""
            <div class="chat-row {bubble_class}-row">
                <div class="chat-bubble {bubble_class}">
                    <div class="bubble-info">
                        <span class="bubble-name">{name}</span>
                        <span class="bubble-time">{msg.get('time', '')}</span>
                    </div>
                    <div class="bubble-content">{msg['content']}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # --- THÊM PHẦN HIỂN THỊ NGUỒN TRÍCH DẪN  ---
        if not is_user and msg.get("sources"):
            st.markdown('<div class="source-divider"></div>', unsafe_allow_html=True)
            with st.expander("📚 Xem nguồn trích dẫn"):
                for i, doc in enumerate(msg["sources"]):
                    page_num = doc.metadata.get('page', 'N/A')
                    if isinstance(page_num, int):
                        page_num += 1 
                        
                    st.markdown(f"**Nguồn {i+1} (Trang {page_num}):**")
                    st.markdown(f"> {doc.page_content}")

        st.markdown("</div></div>", unsafe_allow_html=True)
        
# UPLOAD FILE & USER INPUT -------------------------------------------------------
# user input
user_req = st.chat_input(placeholder="Nhập yêu cầu của bạn...")

# Tạo Key động để Streamlit tự động "rửa sạch" ô upload khi đổi Chat
uploader_key = f"uploader_{st.session_state.current_chat_id}"

# upload file
st.markdown('<div class="chat-input-container-anchor"></div>', unsafe_allow_html=True)
with st.popover("Đính kèm file", use_container_width=False):
    upload_file = st.file_uploader(
        "Upload tài liệu",
        type=["pdf", "docx"],
        label_visibility="collapsed",
        accept_multiple_files=True,
        key=uploader_key
    )

    if upload_file:
        file_key = "_".join([f"{f.name}_{f.size}" for f in upload_file])
        if st.session_state.get("last_file_key") != file_key:
            with st.spinner("Gepity đang xử lý tài liệu..."):
                retriever, vector_store, num_chunks, num_docs = st.session_state.rag.process_document(upload_file)
                st.session_state.retriever = retriever
                st.session_state.vector_store = vector_store
                st.session_state["last_file_key"] = file_key
                st.session_state["file_stats"] = f"Xử lý tài liệu thành công! Số đoạn văn bản: {num_chunks}, Số trang: {num_docs}"
            st.rerun()
    else:
        if st.session_state.get("last_file_key"):
            st.session_state.retriever = None
            st.session_state.vector_store = None
            del st.session_state["last_file_key"]
            if "file_stats" in st.session_state:
                del st.session_state["file_stats"]
                
            st.rerun()

    if "file_stats" in st.session_state:
        st.success(st.session_state["file_stats"])
st.markdown('</div>', unsafe_allow_html=True)

# JAVASCRIPT AUTO-SCROLL -------------------------------------------------------
# only run js when there is message in session, to avoid scroll to bottom when user first open the page
if st.session_state.messages: 
    components.html(f"""
    <script>
        {len(st.session_state.messages)} //new mess -> +1 len -> re-render
        setTimeout(function() {{
            const chatBox = window.parent.document.querySelectorAll(
                '[data-testid="stVerticalBlockBorderWrapper"]'
            )[2];
            
            const userBubbles = window.parent.document.querySelectorAll('.user_bubble');
            const lastUserBubble = userBubbles[userBubbles.length - 1];
            
            if (lastUserBubble && chatBox) {{
                lastUserBubble.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
            }}
        }}, 300);
    </script>
    """, height=0)

if user_req:
    current_time = datetime.now().strftime("%H:%M · %d/%m/%Y")
    st.session_state.messages.append({"role": "user", "content": user_req, "time": current_time})
    with st.spinner("Gepity đang suy nghĩ..."):
        response, sources = st.session_state.rag.get_response(user_req, st.session_state.retriever)
        ai_time = datetime.now().strftime("%H:%M · %d/%m/%Y")
        st.session_state.messages.append({
            "role": "ai", 
            "content": response, 
            "time": ai_time,
            "sources": sources
        })

    st.rerun()