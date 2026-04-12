from sqlalchemy import all_
import streamlit as st
import os
from datetime import datetime 
import streamlit.components.v1 as components
from utils import img_to_base64
from core import RAG_engine, Graph_engine


# SETUP & SESSIONS -------------------------------------------------------
st.set_page_config(page_title="Gepity AI", layout="wide")

if "rag" not in st.session_state:
    st.session_state.rag = RAG_engine()

if "graph_engine" not in st.session_state:
    st.session_state.graph_engine = Graph_engine()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# LOAD CSS -------------------------------------------------------
# Get the directory where app.py actually lives
current_dir = os.path.dirname(os.path.abspath(__file__))
# Build the full path to style.css
css_path = os.path.join(current_dir, "style.css")
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
    st.markdown(f"""
        <div class="logo">
            <div class="logo-ico">G</div>
            <div class="logo-text">Gepity</div>
        </div>
        <div class="logo-info">AI Document Q&A System</div>
        <div class="line"></div>
        <button class="sidebar-button"><div>+</div>Cuộc trò chuyện mới</button>

        <div class="sections">
            <div class="sections-title">LỊCH SỬ TRÒ CHUYỆN</div>
            <div class="sections-list">
                <div class="sections-item sections-item-selected">
                    <img src="data:image/png;base64,{section_ico}" alt="section-ico.png" class="sections-ico" />
                    <span class="section-text">Phân tích báo cáo tài chính</span>
                </div>
                <div class="sections-item">
                    <img src="data:image/png;base64,{section_ico}" alt="section-ico.png" class="sections-ico" />
                    <span class="section-text">Hỏi về hợp đồng lao động</span>
                </div>
                <div class="sections-item">
                    <img src="data:image/png;base64,{section_ico}" alt="section-ico.png" class="sections-ico" />
                    <span class="section-text">Tóm tắt luận văn tốt nghiệp</span>
                </div>
                <div class="sections-item">
                    <img src="data:image/png;base64,{section_ico}" alt="section-ico.png" class="sections-ico" />
                    <span class="section-text">Đề cương môn học OSSD abcdefghijklmnopqrstwxyz</span>
                </div>
            </div>
        </div>
        
        <div class="intructions">
            <div class="intructions-title">HƯỚNG DẪN SỬ DỤNG</div>
            <div class="intructions-item"><div class="item-order">1</div>Upload file PDF hoặc DOCX cần hỏi</div>
            <div class="intructions-item"><div class="item-order">2</div>Chờ hệ thống xử lý tài liệu</div>
            <div class="intructions-item"><div class="item-order">3</div>Nhập câu hỏi về nội dung tài liệu</div>
            <div class="intructions-item"><div class="item-order">4</div>Xem câu trả lời và tiếp tục hỏi</div>
        </div>
        <div class="line"></div>
        <div class="model">
            <div class="model-title">CẤU HÌNH MODEL</div>
            <div class="model-item"><div class="model-title model-item-title">Model đang dùng</div>qwen2.5:7b</div>
            <div class="model-item">
                <div class="model-title model-item-title">
                    Trạng thái
                </div>
                <div class="model-online">
                    <div class="model-online-ico"></div>
                    <div class="model-online-text">Online</div>
                </div>
            </div>
            <div class="model-item"><div class="model-title model-item-title">Embedding</div>multilingual-mpnet</div>
            <div class="model-item"><div class="model-title model-item-title">Vector Store</div>FAISS</div>
        </div>
        <div class="line"></div>
        <button class="sidebar-delete">
            <img src="data:image/png;base64,{trash_can}" alt="delete-ico.png" class="sidebar-delete-ico" />
            Xóa lịch sử chat
        </button>
    """, unsafe_allow_html=True)

# HEADER SECTION -------------------------------------------------------
st.markdown(f"""
    <div class="main-header">
        <div class="header-left">
            <div class="header-left-main">SmartDoc AI — Gepity</div>
            <div class="header-left-sub">Hỏi đáp thông minh từ tài liệu của bạn</div>
        </div>
        <div class="header-right">
            <div class="header-right-circle"></div>
            <div class="header-right-text">qwen2.5:7b</div>
        </div>
    </div>
""", unsafe_allow_html=True)


# CHAT SECTION -------------------------------------------------------
with st.container(height=480):
    chat_placeholder = st.empty()
    
    def redraw_chats():
        with chat_placeholder.container():
            for msg in st.session_state.messages:
                is_user = msg["role"] == 'user'
                css_class = 'user_bubble' if is_user else 'ai_bubble'
                avatar = user_bubble if is_user else ai_bubble
                name = "Bạn" if is_user else "Gepity"
                time_str = msg.get("time", datetime.now().strftime("%H:%M · %d/%m/%Y")) # use current time if not provided

                st.markdown(f"""
                    <div class="bubble-header-{css_class}">
                        <img src="data:image/png;base64,{avatar}" class="{css_class}-ico"/>
                        <div class="bubble-answer-{css_class}">
                            <span class="bubble-name">{name}</span>
                            <div class="{css_class}">{msg["content"]}</div>
                            <span class="bubble-time">{time_str}</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

    redraw_chats()
        
# UPLOAD FILE & USER INPUT -------------------------------------------------------
st.markdown('<div class="uploader-anchor"></div>', unsafe_allow_html=True)

with st.popover("Đính kèm", use_container_width=False):
    #upload file
    upload_file = st.file_uploader(
        label="Upload tài liệu",
        type=["pdf", "docx"],
        help="Hỗ trợ PDF và DOCX",
        label_visibility="collapsed",
        accept_multiple_files=True
    )

    if upload_file:
        file_key = "_".join([f"{f.name}_{f.size}" for f in upload_file])
        if st.session_state.get("last_file_key") != file_key:
            st.session_state["last_file_key"] = file_key

            with st.spinner("Gepity đang xử lý tài liệu..."):
                # basic RAG processing
                retriever, vector_store, num_chunks, num_docs = st.session_state.rag.process_document(upload_file)
                st.success(f"Xử lý tài liệu thành công! Số đoạn văn bản: {num_chunks}, Số trang: {num_docs}")

                # graph RAG processing and sync to graph database
                with st.expander("Chi tiết quá trình xây dựng Graph", expanded=True):
                    all_chunks = st.session_state.graph_engine.process_document(upload_file)
                    nodes_count = st.session_state.graph_engine.sync_to_graph(all_chunks)
                    st.info(f"Đã trích xuất và kết nối các thực thể trên Neo4j. Tổng số nodes: {nodes_count}")
                st.session_state.retriever = retriever
                st.session_state.vector_store = vector_store

# user input
user_req = st.chat_input(placeholder="Nhập yêu cầu của bạn...")

# store response into session
if user_req:
    # add user request to session
    current_time = datetime.now().strftime("%H:%M · %d/%m/%Y")
    st.session_state.messages.append({"role": "user", "content": user_req, "time": current_time})

    redraw_chats()

    # get response from llm with loading spinner
    with st.spinner("Gepity đang suy nghĩ..."):
        # basic RAG response
        basic_response = st.session_state.rag.get_response(user_req, st.session_state.retriever)

        # graph RAG response
        graph_response = st.session_state.graph_engine.get_response(user_req)

        ai_time = datetime.now().strftime("%H:%M · %d/%m/%Y")

        # add llm's response to session
        st.session_state.messages.append({
            "role": "ai", 
            "basic_content": basic_response, 
            "content": graph_response,
            "time": ai_time
        }) 

    st.rerun()


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