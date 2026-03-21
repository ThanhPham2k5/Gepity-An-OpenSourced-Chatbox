import streamlit as st
from langchain_ollama import OllamaLLM
import base64
import os
from datetime import datetime 
import streamlit.components.v1 as components

# setup
st.set_page_config(page_title="Gepity AI", layout="wide")

llm = OllamaLLM(
    # model="qwen2.5:7b",
    model="qwen2.5:3b",
    base_url="http://172.25.64.1:11434"
)

# insert css
def load_css(file_path):
    with open(file_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# convert img to base64
def img_to_base64(path):
    base_dir = os.path.dirname(__file__)
    # print(f"{base_dir}")
    full_path = os.path.join(base_dir, path)
    # print(f"{full_path}")
    with open(full_path, "rb") as f:
        return base64.b64encode(f.read()).decode()
    
trash_can = img_to_base64("assets/img/trash-can.png")
section_ico = img_to_base64("assets/img/section-ico.png")

# sidebar section
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

# header section
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

# get request from user
user_req = st.chat_input(placeholder="Nhập yêu cầu của bạn...")

if "messages" not in st.session_state:
    st.session_state.messages = []

# store response into session
if user_req != None:
    st.session_state.messages.append({"role": "user", "content": user_req})
    response = llm.invoke(user_req)
    st.session_state.messages.append({"role": "ai", "content": response})

# response (entire history)
user_bubble = img_to_base64("assets/img/user-ico.png")
ai_bubble = img_to_base64("assets/img/ai-ico.png")

with st.container(height=356):
    for msg in st.session_state.messages:
        is_user = msg["role"] == 'user'
        css_class = 'user_bubble' if is_user else 'ai_bubble'
        avatar = user_bubble if is_user else ai_bubble
        name = "Bạn" if is_user else "Gepity"
        time_str = datetime.now().strftime("%H:%M · %d/%m/%Y")

        st.markdown(f"""
            <div class="bubble-header-{css_class}">
                <img src="data:image/png;base64,{avatar}" alt="{avatar}.png" class="{css_class}-ico"/>
                <div class="bubble-answer-{css_class}">
                    <span class="bubble-name">{name}</span>
                    <div class="{css_class}">
                        {msg["content"]}
                    </div>
                    <span class="bubble-time">{time_str}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

# uploader file
upload_file = st.file_uploader(
    label="Upload tài liệu",
    type=["pdf", "docx"],
    help="Hỗ trợ PDF và DOCX",
    label_visibility="collapsed",
    accept_multiple_files=True
)

# javascript
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