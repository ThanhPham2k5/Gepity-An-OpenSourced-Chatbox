import os
import re
import tempfile
import nltk
import itertools
from langchain_community.document_loaders import PDFPlumberLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def format_source_text(text):
    # 1. Thêm xuống dòng trước các số thứ tự ở đầu mục (Ví dụ: "1. ", "2. ", "10. ")
    # (?<!\d) đảm bảo nó không cắt nhầm số thập phân như 3.14
    text = re.sub(r'(?<!\d)(\d+\.\s)', r'\n\n\1', text)
    
    # 2. Thêm xuống dòng trước các dấu bullet point (•) hoặc (-)
    text = re.sub(r'([•])\s', r'\n- ', text)
    
    # 3. Dọn dẹp các khoảng trắng/xuống dòng thừa
    text = re.sub(r'\n\s*\n', '\n\n', text)

    # # 4. Xóa thẻ ** nhưng giữ lại nội dung bên trong
    # text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    
    return text.strip()

def get_docs_from_uploaded_files(uploaded_files: list):
    all_docs = []
    
    for uploaded_file in uploaded_files:
        # reset file pointer to the beginning
        uploaded_file.seek(0)

        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        
        try:
            if suffix.lower() == ".pdf":
                loader = PDFPlumberLoader(tmp_file_path)
            elif suffix.lower() in [".docx", ".doc"]:
                loader = Docx2txtLoader(tmp_file_path)
            else:
                continue 

            docs = loader.load()
            
            original_name = uploaded_file.name
            for doc in docs:
                doc.metadata["source"] = original_name
                doc.metadata["filename"] = original_name 
            
            all_docs.extend(docs)
        finally:
            # Clean up the temporary file
            if(os.path.exists(tmp_file_path)):
                os.unlink(tmp_file_path)

    return all_docs

def split_docs_into_chunks(docs: list, chunk_size=1000, chunk_overlap=100):
    # define a text splitter to split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    all_chunks = text_splitter.split_documents(docs)
    return all_chunks

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
def extract_and_link_entities(chunk_text: str, gliner_model):
    lines = chunk_text.split('\n')
    
    sentences = []
    for line in lines:
        if not line.strip():
            continue
            
        # Use NLTK on each line
        line_sentences = nltk.sent_tokenize(line)
        
        for s in line_sentences:
            # If a single sentence is still too long
            # we force-split it by space to ensure we stay under the 384 token limit
            if len(s.split()) > 250:
                words = s.split()
                # Break into chunks of 200 words
                for i in range(0, len(words), 200):
                    sentences.append(" ".join(words[i:i+200]))
            else:
                sentences.append(s)

    # entity labels
    labels = ["Person", "Organization", "Concept", "Algorithm", "Framework", "Dataset", "Identity Number", "Location", "Event"]
    
    # extract entities using gliner
    all_entities_per_sentence = gliner_model.inference(sentences, labels=labels, threshold=0.7, flat_ner=True)

    unique_entities = {}
    relationships = []

    for i, entities in enumerate(all_entities_per_sentence):
        sentence_entities = []

        for ent in entities:
            name = ent['text'].strip().title()
            label = ent['label']

            if name.isdigit(): continue
            
            # Lưu thực thể duy nhất cho toàn chunk
            if name not in unique_entities:
                unique_entities[name] = {
                    "name": name,
                    "label": label,
                    "description": f"Thực thể loại {label} trích xuất từ văn bản.",
                }
            sentence_entities.append(name)

        # Liên kết các thực thể xuất hiện trong cùng 1 câu
        if len(sentence_entities) > 1:
            for src, tgt in itertools.combinations(set(sentence_entities), 2):
                relationships.append({
                    "source": src,
                    "target": tgt,
                    "type": "CO_OCCURRENCE"
                })

    return list(unique_entities.values()), relationships
                
