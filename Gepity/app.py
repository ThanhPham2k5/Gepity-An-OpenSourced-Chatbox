import time

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

if "graph_engine" not in st.session_state:
    st.session_state.graph_engine = Graph_engine()

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

if "is_processing" not in st.session_state:
    st.session_state.is_processing = False

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
    st.radio(
        "Tùy chọn chế độ tìm kiếm:",
        options=["Vector search", "Hybrid search"],
        key="core_search_method",
        horizontal=True
    )
    st.radio(
        "Tùy chọn chế độ truy xuất dữ liệu:",
        options=[
            "NormalRAG", 
            "GraphRAG", 
            "Cả hai"
        ],
        key="rag_architecture",
        horizontal=True
    )

    # Section: Cài đặt Chunk 
    st.markdown('<div class="sidebar-label">Cài đặt băm dữ liệu</div>', unsafe_allow_html=True)
    with st.expander("Tùy chỉnh Chunk Parameters", expanded=False):
        st.markdown("<span style='font-size: 0.85em; color: gray;'>Thiết lập trước khi upload tài liệu</span>", unsafe_allow_html=True)
        
        chunk_size = st.slider(
            "Chunk Size (Kích thước đoạn)", 
            min_value=100, max_value=2000, value=800, step=100,
            help="Số lượng ký tự tối đa trong một đoạn văn bản. Càng lớn thì ngữ cảnh càng rộng nhưng tốn RAM.",
            disabled=st.session_state.is_processing
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
            help=f"Số ký tự được giữ lại. Hiện tại tối đa là {safe_max_overlap} (50% của Chunk Size).",
            disabled=st.session_state.is_processing
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
        time_str = msg.get('time', '')

        # -----------------------------------------------------
        # 1. NẾU LÀ NGƯỜI DÙNG: Luôn vẽ 1 cột bên phải
        # -----------------------------------------------------
        if is_user:
            st.markdown(f"""
                <div class="bubble-header-user_bubble">
                    <img src="data:image/png;base64,{user_bubble}" class="user_bubble-ico"/>
                    <div class="bubble-answer-user_bubble">
                        <span class="bubble-name">Bạn</span>
                        <div class="user_bubble">{msg["content"]}</div>
                        <span class="bubble-time">{time_str}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # -----------------------------------------------------
        # 2. NẾU LÀ AI: Quyết định vẽ 1 hay 2 cột dựa vào cờ is_compare
        # -----------------------------------------------------
        else:
            if msg.get("is_compare"):
                # === TRƯỜNG HỢP CHIA ĐÔI MÀN HÌNH ===
                col1, col2 = st.columns(2)
                
                with col1: # CỘT TRÁI (HYBRID RAG - Luôn là Document Object)
                    st.markdown(f"""
                        <div class="bubble-header-ai_bubble">
                            <div class="bubble-answer-ai_bubble">
                                <span class="bubble-name">{msg['left_title']}</span>
                                <div class="ai_bubble">{msg["left_content"]}</div>
                                <span class="bubble-time">{time_str}</span>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    if msg.get("left_sources"):
                        with st.expander("Nguồn (Hybrid RAG)"):
                            for i, doc in enumerate(msg["left_sources"]):
                                p = doc.metadata.get('page', 'N/A')
                                p = p + 1 if isinstance(p, int) else p
                                f_name = doc.metadata.get('file_name', 'Tài liệu')
                                st.markdown(f"**{f_name} (Trang {p})**")
                                st.markdown(f"> {doc.page_content}")
                                
                with col2: # CỘT PHẢI (GRAPH RAG - Luôn là Dictionary)
                    st.markdown(f"""
                        <div class="bubble-header-ai_bubble">
                            <div class="bubble-answer-ai_bubble">
                                <span class="bubble-name">{msg['right_title']}</span>
                                <div class="ai_bubble">{msg["right_content"]}</div>
                                <span class="bubble-time">{time_str}</span>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    if msg.get("right_sources"):
                        with st.expander("Nguồn (GraphRAG)"):
                            for i, doc in enumerate(msg["right_sources"]):
                                # Lấy dữ liệu từ Dictionary của GraphRAG
                                p = doc.get('page_number', 'N/A')
                                f_name = doc.get('file_name', 'Tài liệu')
                                score = doc.get('score', 0)
                                content = doc.get('text', '')
                                
                                st.markdown(f"**{f_name} (Trang {p} - Độ tin cậy: {score:.2f})**")
                                st.markdown(f"> {content}")

            else:
                # === TRƯỜNG HỢP 1 CỘT (CHẾ ĐỘ ĐƠN LẺ) ===
                mode_label = f" - {msg.get('mode_name', '')}" if msg.get('mode_name') else ""
                st.markdown(f"""
                    <div class="bubble-header-ai_bubble">
                        <img src="data:image/png;base64,{ai_bubble}" class="ai_bubble-ico"/>
                        <div class="bubble-answer-ai_bubble">
                            <span class="bubble-name">Gepity{mode_label}</span>
                            <div class="ai_bubble">{msg["content"]}</div>
                            <span class="bubble-time">{time_str}</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                if msg.get("sources"):
                    st.markdown('<div class="source-divider"></div>', unsafe_allow_html=True)
                    with st.expander("Nguồn trích dẫn"):
                        for i, doc in enumerate(msg["sources"]):
                            # --- PHÉP THUẬT NẰM Ở ĐÂY: Tự động phân loại dữ liệu ---
                            if isinstance(doc, dict):
                                # Nếu là GraphRAG (Dictionary)
                                p = doc.get('page_number', 'N/A')
                                f_name = doc.get('file_name', 'Tài liệu')
                                content = doc.get('text', '')
                                score_text = f" - Score: {doc.get('score', 0):.2f}"
                            else:
                                # Nếu là Vector RAG (Document Object)
                                p = doc.metadata.get('page', 'N/A')
                                p = p + 1 if isinstance(p, int) else p
                                f_name = doc.metadata.get('file_name', 'Tài liệu')
                                content = doc.page_content
                                score_text = ""
                            
                            st.markdown(f"**Nguồn {i+1} ({f_name} - Trang {p}{score_text}):**")
                            st.markdown(f"> {content}")
        
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

    # Đảm bảo string mặc định khớp với các trường hợp bên dưới
    rag_arch = st.session_state.get("rag_architecture", "NormalRAG")
    
    # Cờ báo hiệu xem có cần reload lại trang không
    needs_rerun = False 

    if upload_file:
        file_key = "_".join([f"{f.name}_{f.size}" for f in upload_file])
        st.session_state.is_processing = True
        # ========================================================
        # LUỒNG 1: VECTOR RAG (Bán cầu Trái)
        # ========================================================
        if "Normal" in rag_arch or "Cả hai" in rag_arch:
            if st.session_state.get("last_vector_key") != file_key:
                with st.status("Gepity đang nạp kiến thức Vector...", expanded=True) as status:
                    for f in upload_file: f.seek(0)

                    start_time = time.time()

                    retriever, vector_store, num_chunks, num_docs = st.session_state.rag.process_document(
                        upload_file, 
                        chunk_size=st.session_state.get("chunk_size", 500), 
                        chunk_overlap=st.session_state.get("chunk_overlap", 50)
                    )

                    end_time = time.time()

                    st.session_state.retriever = retriever
                    st.session_state.vector_store = vector_store
                    st.session_state["last_vector_key"] = file_key
                    st.session_state["uploaded_filenames"] = [f.name for f in upload_file]
                    st.session_state["file_stats"] = f"Xử lý tài liệu thành công! Tổng số chunk: {num_chunks}, Thời gian: {end_time - start_time:.2f} giây"
                    
                    status.update(label=f"Hoàn tất Vector DB! ({num_chunks} đoạn)", state="complete", expanded=False)
                    needs_rerun = True

        # ========================================================
        # LUỒNG 2: GRAPH RAG (Bán cầu Phải)
        # ========================================================
        if "Graph" in rag_arch or "Cả hai" in rag_arch:
            if st.session_state.get("last_graph_key") != file_key:        
                with st.status("Gepity đang rút trích Graph cho Neo4j (Sẽ mất thời gian)...", expanded=True) as status:
                    for f in upload_file: f.seek(0)

                    start_time = time.time()

                    docs_with_chunks = st.session_state.graph_engine.process_document(
                        uploaded_files=upload_file,
                        chunk_size=st.session_state.get("chunk_size", 800), 
                        chunk_overlap=st.session_state.get("chunk_overlap", 80)
                    )

                    total_processed_chunks = 0
                    # process each file from multiple files
                    for filename, chunks in docs_with_chunks.items():
                        st.write(f"Đang xử lý tài liệu: {filename}")
                        
                        st.session_state.graph_engine.sync_to_graph(chunks, source=filename)
                        
                        total_processed_chunks += len(chunks)
                    
                    st.session_state.graph_engine.create_vector_indexes()
                    st.session_state.graph_engine.create_fulltext_index()
                    end_time = time.time()

                    st.session_state["last_graph_key"] = file_key
                    st.session_state["file_stats"] = f"Đã trích xuất và kết nối các thực thể trên Neo4j! Tổng số chunk: {total_processed_chunks}, Thời gian: {end_time - start_time:.2f} giây"
                    status.update(label=f"Hoàn tất nạp Neo4j ({total_processed_chunks} chunks)!", state="complete", expanded=False)

                    needs_rerun = True

        # ========================================================
        # LOAD LẠI TRANG (Nếu có xử lý file mới)
        # ==========================================
        if needs_rerun:
            st.rerun()

    else:
        # ========================================================
        # DỌN RÁC HIỆN TRƯỜNG KHI NGƯỜI DÙNG XÓA FILE
        # ========================================================
        has_vector = "last_vector_key" in st.session_state
        has_graph = "last_graph_key" in st.session_state
        
        # Nếu phát hiện còn file cũ trong bộ nhớ thì mới dọn dẹp
        if has_vector or has_graph:
            st.session_state.retriever = None
            st.session_state.vector_store = None
            
            if has_vector: del st.session_state["last_vector_key"]
            if has_graph: del st.session_state["last_graph_key"]
            if "file_stats" in st.session_state: del st.session_state["file_stats"]
            if "uploaded_filenames" in st.session_state: del st.session_state["uploaded_filenames"]
            
            st.rerun()

    # Hiển thị thông báo thành công dưới khung Upload
    if "file_stats" in st.session_state:
        st.success(st.session_state["file_stats"])

st.markdown('</div>', unsafe_allow_html=True)

if user_req:
    current_time = datetime.now().strftime("%H:%M · %d/%m/%Y")
    st.session_state.messages.append({"role": "user", "content": user_req, "time": current_time})
    
    with st.spinner("Gepity đang suy nghĩ..."):
        ai_time = datetime.now().strftime("%H:%M · %d/%m/%Y")
        core_method = st.session_state.get("core_search_method", "Hybrid search")
        rag_arch = st.session_state.get("rag_architecture", "NormalRAG")

        vector_search_kwargs = {"k": 3} 
        if filter_choice and filter_choice != "Tất cả":
            vector_search_kwargs["filter"] = {"file_name": filter_choice}

        pure_vector_retriever = None
        if st.session_state.vector_store:
            pure_vector_retriever = st.session_state.vector_store.as_retriever(search_kwargs=vector_search_kwargs)

        if st.session_state.retriever:
            if hasattr(st.session_state.retriever, "retrievers"): 
                for sub_retriever in st.session_state.retriever.retrievers:
                    if hasattr(sub_retriever, "search_kwargs"): 
                        sub_retriever.search_kwargs = vector_search_kwargs
                    elif hasattr(sub_retriever, "k"): 
                        sub_retriever.k = 15
            elif hasattr(st.session_state.retriever, "search_kwargs"):
                st.session_state.retriever.search_kwargs = vector_search_kwargs

        active_retriever = st.session_state.retriever if core_method == "Hybrid search" else pure_vector_retriever
        normal_rag_name = f"NormalRAG ({core_method})"

        if "Cả hai" in rag_arch:
            res_normal, sources_normal = st.session_state.rag.get_response(
                user_req, retriever=active_retriever, filter_filename=filter_choice, chat_history=st.session_state.messages
            )
            res_graph, sources_graph = st.session_state.graph_engine.get_response(user_req)
            
            st.session_state.messages.append({
                "role": "ai", "is_compare": True, 
                "left_title": normal_rag_name,
                "left_content": res_normal, "left_sources": sources_normal,
                "right_title": "GraphRAG",
                "right_content": res_graph, "right_sources": sources_graph,
                "time": ai_time
            })

        elif "Graph" in rag_arch:
            res_graph, sources_graph = st.session_state.graph_engine.get_response(user_req)
            st.session_state.messages.append({
                "role": "ai", "is_compare": False, 
                "content": res_graph, "sources": sources_graph, 
                "time": ai_time, "mode_name": "GraphRAG"
            })

        else:
            res_normal, sources_normal = st.session_state.rag.get_response(
                user_req, retriever=active_retriever, filter_filename=filter_choice, chat_history=st.session_state.messages
            )
            st.session_state.messages.append({
                "role": "ai", "is_compare": False, 
                "content": res_normal, "sources": sources_normal, 
                "time": ai_time, "mode_name": normal_rag_name
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