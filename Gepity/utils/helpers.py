import os
import base64
import streamlit as st
from datetime import datetime 

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
                    chat['retriever'] = st.session_state.get('retriever')
                    chat['vector_store'] = st.session_state.get('vector_store')
                    chat['last_file_key'] = st.session_state.get('last_file_key')
                    chat['uploaded_filenames'] = st.session_state.get('uploaded_filenames', [])
                    chat['file_uploader_key'] = st.session_state.get('file_uploader_key', None)
                    break
        # TRƯỜNG HỢP 2: Đang chat ở "Cuộc trò chuyện mới" chưa có ID -> Tạo mới vào lịch sử
        else:
            new_id = datetime.now().strftime("%Y%m%d%H%M%S")
            new_item = {
                "id": new_id,
                "title": title,
                "messages": st.session_state.messages.copy(),
                "retriever": st.session_state.get('retriever'),
                "vector_store": st.session_state.get('vector_store'),
                "last_file_key": st.session_state.get('last_file_key'),
                "uploaded_filenames": st.session_state.get('uploaded_filenames', []),
                "file_uploader_key": st.session_state.get('file_uploader_key', None)
            }
            st.session_state.chat_history.insert(0, new_item)
            st.session_state.current_chat_id = new_id