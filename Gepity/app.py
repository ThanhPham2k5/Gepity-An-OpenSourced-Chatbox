from sqlalchemy import all_
import streamlit as st
import os
from datetime import datetime 
import streamlit.components.v1 as components
from utils.helpers import img_to_base64, update_current_chat_to_history
from core import RAG_engine, Graph_engine
import uuid

# SETUP & SESSIONS -------------------------------------------------------
st.set_page_config(page_title="Gepity AI", layout="wide")
@st.dialog("⚠️ Xác nhận xóa dữ liệu")
def confirm_action_dialog(action_type):
    if action_type == "history":
        st.write("Bạn có chắc chắn muốn xóa **TOÀN BỘ** lịch sử trò chuyện? Hành động này không thể hoàn tác.")
        if st.button("Xác nhận xóa sạch lịch sử", type="primary", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.session_state.current_chat_id = None
            st.rerun()
            
    elif action_type == "vector":
        st.write("Hệ thống sẽ xóa toàn bộ tài liệu đã được băm nhỏ trong bộ nhớ Vector (FAISS).")
        if st.button("Xác nhận giải phóng bộ nhớ", type="primary", use_container_width=True):
            st.session_state.retriever = None
            st.session_state.vector_store = None
            st.session_state.last_file_key = None
            st.session_state.uploaded_filenames = []
            st.session_state.file_uploader_key = str(uuid.uuid4())
            
            if "file_stats" in st.session_state:
                del st.session_state["file_stats"]

            for chat in st.session_state.chat_history:
                chat['retriever'] = None
                chat['vector_store'] = None
                chat['last_file_key'] = None
                chat['uploaded_filenames'] = []
            
            st.rerun()

if "rag" not in st.session_state:
    st.session_state.rag = RAG_engine()

# if "graph_engine" not in st.session_state:
#     st.session_state.graph_engine = Graph_engine()

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

if "file_uploader_key" not in st.session_state:
    st.session_state.file_uploader_key = str(uuid.uuid4())

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
        st.session_state.file_uploader_key = str(uuid.uuid4())
        if "last_file_key" in st.session_state:
            del st.session_state["last_file_key"]
        if "file_stats" in st.session_state:
            del st.session_state["file_stats"]
        if "uploaded_filenames" in st.session_state:
            del st.session_state["uploaded_filenames"]
        st.rerun()

    # Comparison Mode Toggle
    st.toggle("So sánh vector search với hybrid search", key="comparison_mode")

    # Section: Cài đặt Chunk 
    st.markdown('<div class="sidebar-label">Cài đặt băm dữ liệu</div>', unsafe_allow_html=True)
    with st.expander("Tùy chỉnh Chunk Parameters", expanded=False):
        st.markdown("<span style='font-size: 0.85em; color: gray;'>Thiết lập trước khi upload tài liệu</span>", unsafe_allow_html=True)
        
        chunk_size = st.slider(
            "Chunk Size (Kích thước đoạn)", 
            min_value=100, max_value=2000, value=800, step=100,
            help="Số lượng ký tự tối đa trong một đoạn văn bản. Càng lớn thì ngữ cảnh càng rộng nhưng tốn RAM."
        )
        
        safe_max_overlap = int(chunk_size / 2)
        current_overlap = st.session_state.get("chunk_overlap", 50)
        safe_default_overlap = min(current_overlap, safe_max_overlap)

        chunk_overlap = st.slider(
            "Chunk Overlap (Độ trùng lặp)", 
            min_value=0, 
            max_value=safe_max_overlap,
            value=safe_default_overlap, 
            step=10,
            help=f"Số ký tự được giữ lại. Hiện tại tối đa là {safe_max_overlap} (50% của Chunk Size)."
        )

        st.session_state.chunk_size = chunk_size
        st.session_state.chunk_overlap = chunk_overlap

    # Lịch sử trò chuyện
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
                st.session_state['uploaded_filenames'] = chat.get('uploaded_filenames', [])
                st.session_state['file_uploader_key'] = chat.get('file_uploader_key') or str(uuid.uuid4())
                if "file_stats" in st.session_state:
                    del st.session_state["file_stats"]

                st.rerun()

    # Bộ lọc tài liệu
    st.markdown('<div class="sidebar-label">Lọc tài liệu</div>', unsafe_allow_html=True)
    filter_choice = None 
    if "uploaded_filenames" in st.session_state and st.session_state["uploaded_filenames"]:
        danh_sach_file = ["Tất cả"] + list(st.session_state["uploaded_filenames"])
        filter_choice = st.selectbox(
            "Chỉ tìm kiếm trong:",
            options=danh_sach_file,
            label_visibility="collapsed"
        )
    else:
        st.info("Chưa có file nào.")
    
    # Dọn dẹp dữ liệu
    st.divider()
    st.markdown('<div class="sidebar-label">Quản trị dữ liệu</div>', unsafe_allow_html=True)
    
    col_del_1, col_del_2 = st.columns(2)
    with col_del_1:
        if st.button("🗑️ Xóa Chat", use_container_width=True):
            confirm_action_dialog("history")
    with col_del_2:
        if st.button("📂 Xóa Docs", use_container_width=True):
            confirm_action_dialog("vector")

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
        is_user = msg["role"] == 'user'
        css_class = 'user_bubble' if is_user else 'ai_bubble'
        avatar = user_bubble if is_user else ai_bubble
        name = "Bạn" if is_user else "Gepity"
        time_str = datetime.now().strftime("%H:%M · %d/%m/%Y")
        if is_user or not st.session_state.get("comparison_mode", False):
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

            # --- NGUỒN TRÍCH DẪN (STANDARD RAG) ---
            if not is_user and msg.get("sources"):
                st.markdown('<div class="source-divider"></div>', unsafe_allow_html=True)
                with st.expander("📚 Nguồn trích dẫn"):
                    for i, doc in enumerate(msg["sources"]):
                        file_name = doc.metadata.get('file_name', 'Tài liệu')
                        page_num = doc.metadata.get('page', 'N/A')
                        if isinstance(page_num, int):
                            page_num += 1 
                            
                        st.markdown(f"**Nguồn {i+1} ({file_name} - Trang {page_num}):**")
                        st.markdown(f"> {doc.page_content}")

        else:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                    <div class="chat-row ai-bubble-row">
                        <div class="chat-bubble ai-bubble">
                            <div class="bubble-info">
                                <span class="bubble-name">📚 Standard RAG</span>
                                <span class="bubble-time">{msg.get('time', '')}</span>
                            </div>
                            <div class="bubble-content">{msg['content']}</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                if msg.get("sources"):
                    with st.expander("📚 Nguồn trích dẫn"):
                        for i, doc in enumerate(msg["sources"]):
                            file_name = doc.metadata.get('file_name', 'Tài liệu')
                            page_num = doc.metadata.get('page', 'N/A')
                            page_num = page_num + 1 if isinstance(page_num, int) else page_num
                            st.markdown(f"**{file_name} (Trang {page_num})**")
                            st.markdown(f"> {doc.page_content}")

            with col2:
                hybrid_content = msg.get("hybrid_content")
                if hybrid_content:
                    st.markdown(f"""
                        <div class="chat-row ai-bubble-row">
                            <div class="chat-bubble ai-bubble">
                                <div class="bubble-info">
                                    <span class="bubble-name">🧠 Hybrid RAG (Vector + BM25)</span>
                                    <span class="bubble-time">{msg.get('time', '')}</span>
                                </div>
                                <div class="bubble-content">{hybrid_content}</div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                
                    if msg.get("hybrid_sources"):
                        with st.expander("📚 Nguồn trích dẫn"):
                            for i, doc in enumerate(msg["hybrid_sources"]):
                                file_name = doc.metadata.get('file_name', 'Tài liệu')
                                page_num = doc.metadata.get('page', 'N/A')
                                page_num = page_num + 1 if isinstance(page_num, int) else page_num
                                st.markdown(f"**{file_name} (Trang {page_num})**")
                                st.markdown(f"> {doc.page_content}")
        
# UPLOAD FILE & USER INPUT -------------------------------------------------------
# user input
user_req = st.chat_input(placeholder="Nhập yêu cầu của bạn...")

# upload file
st.markdown('<div class="chat-input-container-anchor"></div>', unsafe_allow_html=True)
with st.popover("Đính kèm file", use_container_width=False):
    upload_file = st.file_uploader(
        "Upload tài liệu",
        type=["pdf", "docx"],
        label_visibility="collapsed",
        accept_multiple_files=True,
        key=st.session_state.file_uploader_key
    )

    if upload_file:
        file_key = "_".join([f"{f.name}_{f.size}" for f in upload_file])
        if st.session_state.get("last_file_key") != file_key:
            with st.spinner("Gepity đang xử lý tài liệu..."):

                # basic rag
                # retriever, vector_store, num_chunks, num_docs = st.session_state.rag.process_document(
                #     upload_file, 
                #     chunk_size=st.session_state.get("chunk_size", 500), 
                #     chunk_overlap=st.session_state.get("chunk_overlap", 50)
                # )
                # st.session_state.retriever = retriever
                # st.session_state.vector_store = vector_store
                # st.session_state["last_file_key"] = file_key
                # st.session_state["uploaded_filenames"] = [f.name for f in upload_file]
                # st.session_state["file_stats"] = f"Xử lý tài liệu thành công! Số đoạn văn bản: {num_chunks}, Số trang: {num_docs}"

                # graph rag
                with st.expander("Chi tiết quá trình xây dựng Graph", expanded=True):
                    chunks = st.session_state.graph_engine.process_document(
                        uploaded_files=upload_file,
                        chunk_size=st.session_state.get("chunk_size", 800), 
                        chunk_overlap=st.session_state.get("chunk_overlap", 80)
                    )
                    chunk_count = st.session_state.graph_engine.sync_to_graph(chunks)
                    st.info(f"Đã trích xuất và kết nối các thực thể trên Neo4j. Tổng số chunk: {chunk_count}")




            st.rerun()
    else:
        current_key = st.session_state.get("last_file_key")
        
        if current_key:
            st.session_state.retriever = None
            st.session_state.vector_store = None
            del st.session_state["last_file_key"]
            if "file_stats" in st.session_state:
                del st.session_state["file_stats"]
                
            st.rerun()

    if "file_stats" in st.session_state:
        st.success(st.session_state["file_stats"])
st.markdown('</div>', unsafe_allow_html=True)

if user_req:
    current_time = datetime.now().strftime("%H:%M · %d/%m/%Y")
    st.session_state.messages.append({"role": "user", "content": user_req, "time": current_time})
    
    with st.spinner("Gepity đang suy nghĩ..."):
        ai_time = datetime.now().strftime("%H:%M · %d/%m/%Y")

        vector_search_kwargs = {"k": 3} 
        if filter_choice and filter_choice != "Tất cả":
            vector_search_kwargs["filter"] = {"file_name": filter_choice}

        pure_vector_retriever = None
        if st.session_state.vector_store:
            pure_vector_retriever = st.session_state.vector_store.as_retriever(search_kwargs=vector_search_kwargs)

        if st.session_state.retriever:
            for sub_retriever in st.session_state.retriever.retrievers:
                if hasattr(sub_retriever, "search_kwargs"): 
                    sub_retriever.search_kwargs = vector_search_kwargs
                elif hasattr(sub_retriever, "k"): 
                    sub_retriever.k = 15

        if st.session_state.get("comparison_mode", False):
            
            res_standard, sources_standard = st.session_state.rag.get_response(
                user_req, pure_vector_retriever, filter_filename=filter_choice, chat_history=st.session_state.messages
            )
            res_hybrid, sources_hybrid = st.session_state.rag.get_response(
                user_req, st.session_state.retriever, filter_filename=filter_choice, chat_history=st.session_state.messages
            )
            
            st.session_state.messages.append({
                "role": "ai",
                "content": res_standard,
                "sources": sources_standard,
                "hybrid_content": res_hybrid,
                "hybrid_sources": sources_hybrid,
                "time": ai_time
            })
            
        else:
            response, sources = st.session_state.rag.get_response(
                user_req, st.session_state.retriever, filter_filename=filter_choice, chat_history=st.session_state.messages
            )
            st.session_state.messages.append({"role": "ai", "content": response, "sources": sources, "time": ai_time})

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