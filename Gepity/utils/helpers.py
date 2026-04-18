import os
import base64
import streamlit as st
from datetime import datetime 
import re
import time
import json

def img_to_base64(path):
    # get current directory of this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # get project root by going up one level from current directory
    project_root = os.path.dirname(current_dir)
    
    # construct full path to the image file
    full_path = os.path.join(project_root, path)
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Không tìm thấy file tại: {full_path}")
        
    with open(full_path, "rb") as f:
        return base64.b64encode(f.read()).decode()
    

def is_vietnamese(text: str) -> bool:
    vn_chars = "àáâãèéêìíòóôõùúýăđơưạảấầẩẫậắặẳẵặẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ"
    return any(c in text.lower() for c in vn_chars)

def setup_constraints(graph):
    constraints_cypher = [
        "CREATE CONSTRAINT document_source_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.source IS UNIQUE;",
        "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE;",
        "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE;"
    ]
    for cypher in constraints_cypher:
        graph.query(cypher)

def extract_json_from_response(content):
    """Sử dụng Regex để tìm và bóc tách khối JSON từ phản hồi của LLM."""
    # Tìm kiếm khối bắt đầu bằng { và kết thúc bằng }
    match = re.search(r'\{.*\}', content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    
    # Fallback nếu không dùng regex được
    clean_content = content.replace("```json", "").replace("```", "").strip()
    return json.loads(clean_content)


def update_current_chat_to_history():
    if st.session_state.messages:
        first_q = st.session_state.messages[0]['content']
        title = (first_q[:20] + '...') if len(first_q) > 20 else first_q
        
        # TRƯỜNG HỢP 1: Đang ở một cuộc trò chuyện cũ (đã có ID) -> Cập nhật lại nó
        if st.session_state.current_chat_id:
            for chat in st.session_state.chat_history:
                if chat['id'] == st.session_state.current_chat_id:
                    chat['messages'] = st.session_state.messages.copy()
                    chat['title'] = title
                    break
        # TRƯỜNG HỢP 2: Đang chat ở "Cuộc trò chuyện mới" chưa có ID -> Tạo mới vào lịch sử
        else:
            new_id = datetime.now().strftime("%Y%m%d%H%M%S")
            new_item = {
                "id": new_id,
                "title": title,
                "messages": st.session_state.messages.copy()
            }
            st.session_state.chat_history.insert(0, new_item)
            st.session_state.current_chat_id = new_id
