import os
import base64

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