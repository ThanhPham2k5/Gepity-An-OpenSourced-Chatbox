import streamlit as st
from langchain_ollama import OllamaLLM
import base64
import os
from datetime import datetime 
import streamlit.components.v1 as components
import tempfile
import time

from langchain_community.document_loaders import PDFPlumberLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate


# setup -------------------------------------------------------
st.set_page_config(page_title="Gepity AI", layout="wide")

llm = OllamaLLM(
    # model="qwen2.5:7b",
    model="qwen2.5:3b",
    base_url="http://localhost:11434"
)

# insert css -------------------------------------------------------
def load_css(file_path):
    with open(file_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# convert img to base64 -------------------------------------------------------
def img_to_base64(path):
    base_dir = os.path.dirname(__file__)
    # print(f"{base_dir}")
    full_path = os.path.join(base_dir, path)
    # print(f"{full_path}")
    with open(full_path, "rb") as f:
        return base64.b64encode(f.read()).decode()
    
trash_can = img_to_base64("assets/img/trash-can.png")
section_ico = img_to_base64("assets/img/section-ico.png")


if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# embedder -------------------------------------------------------
def get_embedder():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

embedder = get_embedder()

# load document -------------------------------------------------------
def load_document(uploaded_files: list):

    all_docs = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        
        loader = PDFPlumberLoader(tmp_file_path)
        docs = loader.load()
        os.unlink(tmp_file_path)
        all_docs.extend(docs)

    # text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )

    documents = text_splitter.split_documents(all_docs)

    vector_store = FAISS.from_documents(documents, embedder)

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    return retriever, vector_store, len(documents), len(all_docs)

# ─── LANGUAGE DETECTION ─────────────────────────────────────────────────────────
def is_vietnamese(text: str) -> bool:
    vn_chars = "àáâãèéêìíòóôõùúýăđơưạảấầẩẫậắặẳẵặẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ"
    return any(c in text.lower() for c in vn_chars)

# build prompt -------------------------------------------------------
def build_prompt(context: str, user_input: str) -> str:
    if is_vietnamese(user_input):
        return f""" Sử dụng ngữ cảnh sau đây để trả lời câu hỏi.
        Nếu bạn không biết, chỉ cần nói là bạn không biết.
        Trả lời ngắn gọn (3-4 câu) BẮT BUỘC bằng tiếng Việt.
        
        Ngữ cảnh: {context}
        
        Câu hỏi: {user_input}
        
        Trả lời:"""
    else:
        return f"""Use the following context to answer the question.
        If you don't know the answer, just say you don't know.
        Keep answer concise (3-4 sentences).
        
        Context: {context}
        
        Question: {user_input}
        
        Answer:"""


def rag_query(user_input: str) -> str:
    retriever = st.session_state.retriever
    if retriever is None:
        return llm.invoke(user_input)
    # query embedding + similarity search
    relevant_docs = retriever.invoke(user_input)
    # Context building
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # build prompt
    prompt = build_prompt(context, user_input)

    # llm response
    response = llm.invoke(prompt)

    return response

# sidebar section -------------------------------------------------------
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

# header section -------------------------------------------------------
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

# get request from user -------------------------------------------------------
user_req = st.chat_input(placeholder="Nhập yêu cầu của bạn...")

# response (entire history) -------------------------------------------------------
user_bubble = img_to_base64("assets/img/user-ico.png")
ai_bubble = img_to_base64("assets/img/ai-ico.png")

with st.container(height=480):
    chat_placeholder = st.empty()
    
    def redraw_chats():
        with chat_placeholder.container():
            for msg in st.session_state.messages:
                is_user = msg["role"] == 'user'
                css_class = 'user_bubble' if is_user else 'ai_bubble'
                avatar = user_bubble if is_user else ai_bubble
                name = "Bạn" if is_user else "Gepity"
                time_str = datetime.now().strftime("%H:%M · %d/%m/%Y")

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
        
# store response into session -------------------------------------------------------
if user_req != None:
    st.session_state.messages.append({"role": "user", "content": user_req})

    redraw_chats()

    with st.spinner("Gepity đang suy nghĩ..."):
        response = rag_query(user_req)
        st.session_state.messages.append({"role": "ai", "content": response})

    st.rerun()

# uploader file -------------------------------------------------------
st.markdown('<div class="uploader-anchor"></div>', unsafe_allow_html=True)

with st.popover("Đính kèm", use_container_width=False):
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
                retriever, vector_store, num_chunks, num_docs = load_document(upload_file)
                st.session_state.retriever = retriever
                st.session_state.vector_store = vector_store
            
            st.success(f"Xử lý tài liệu thành công! Số đoạn văn bản: {num_chunks}, Số trang: {num_docs}")

# javascript -------------------------------------------------------
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