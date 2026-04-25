import os
import json
import streamlit as st
import re
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from .processor import get_docs_from_uploaded_files, split_docs_into_chunks
from langchain_community.vectorstores import FAISS
from utils import is_vietnamese
from datetime import datetime
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.ai import AIMessage
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
env_path = ROOT_DIR / '.env'
load_dotenv(dotenv_path=env_path)

# Use for Windows IP
#WINDOWS_IP = "172.25.64.1"

# Use for WSL IP
WINDOWS_IP = os.getenv("WINDOWS_IP", "localhost")
MODELS_CACHE = os.getenv("MODELS_CACHE", "../models_cache")

SAVE_DIR = "gepity_database"
META_FILE = f"{SAVE_DIR}/metadata.json"

class RAG_engine:
    def __init__(self, model_name="qwen2.5:3b"):
        self.llm = OllamaLLM(model=model_name, base_url=f"http://{WINDOWS_IP}:11434")
        # This tells the library where to save the model on your disk
        os.environ['HF_HOME'] = MODELS_CACHE
        self.embedder = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            model_kwargs={"device": "cpu", "token": os.getenv("HF_TOKEN")},
            encode_kwargs={"normalize_embeddings": True},
            cache_folder= MODELS_CACHE
        )

    def process_document(self, uploaded_files, chunk_size=500, chunk_overlap=50, existing_vector_store=None):
        all_docs = []
        current_date = datetime.now().strftime("%d/%m/%Y")
        for file in uploaded_files:
            docs_of_this_file = get_docs_from_uploaded_files([file])
            
            for doc in docs_of_this_file:
                doc.metadata['file_name'] = file.name 
                doc.metadata['upload_date'] = current_date
                doc.metadata['file_type'] = "pdf" if file.name.lower().endswith('.pdf') else "docx"
            
            all_docs.extend(docs_of_this_file)

        all_chunks = split_docs_into_chunks(all_docs, chunk_size, chunk_overlap)

        if existing_vector_store:
            existing_vector_store.add_documents(all_chunks)
            vector_store = existing_vector_store
        else:
            vector_store = FAISS.from_documents(all_chunks, self.embedder)

        faiss_retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        all_global_docs = list(vector_store.docstore._dict.values())

        bm25_retriever = BM25Retriever.from_documents(all_global_docs)
        bm25_retriever.k = 3

        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever], 
            weights=[0.5, 0.5]
        )

        return ensemble_retriever, vector_store, len(all_chunks), len(all_docs)
       
    def get_response(self, user_input, retriever, filter_filename=None, chat_history=None):
        # Xử lý lịch sử trò chuyện
        formatted_history = []
        if chat_history:
            for msg in chat_history:
                if msg["role"] == "user":
                    formatted_history.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "ai":
                    if msg.get("is_compare"):
                        combined_content = f"Câu trả lời Normal RAG: {msg.get('left_content', '')}\nCâu trả lời GraphRAG: {msg.get('right_content', '')}"
                        formatted_history.append(AIMessage(content=combined_content))
                    else:
                        formatted_history.append(AIMessage(content=msg.get("content", "")))

        if retriever is None:
            if formatted_history:
                prompt_chitchat = ChatPromptTemplate.from_messages([
                    ("system", "Bạn là một trợ lý AI hữu ích."),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}"),
                ])
                chain_chitchat = prompt_chitchat | self.llm
                result = chain_chitchat.invoke({"input": user_input, "chat_history": formatted_history})
                answer = result.content if hasattr(result, "content") else result
                return answer, None
            else:
                result = self.llm.invoke(user_input)
                answer = result.content if hasattr(result, "content") else result
                return answer, None

        intent_system_prompt = (
            "Bạn là một bộ phân loại ý định (Intent Classifier). Nhiệm vụ của bạn là phân loại câu hỏi thành 1 trong 2 loại:\n"
            "1. 'CHITCHAT': CHỈ dành cho các câu giao tiếp xã giao thuần túy (VD: Xin chào, cảm ơn, tạm biệt).\n"
            "2. 'DOC_SEARCH': Tất cả các câu hỏi còn lại. Bao gồm hỏi về thông tin, yêu cầu, công việc, tính năng, hoặc bất cứ câu hỏi nào có khả năng cần tra cứu.\n\n"
            "NẾU PHÂN VÂN, HÃY LUÔN CHỌN 'DOC_SEARCH'. Chỉ trả về ĐÚNG 1 TỪ duy nhất: 'CHITCHAT' hoặc 'DOC_SEARCH'."
        )
        intent_prompt = ChatPromptTemplate.from_messages([
            ("system", intent_system_prompt),
            ("human", "{input}"),
        ])
        
        intent_chain = intent_prompt | self.llm
        intent_result = intent_chain.invoke({"input": user_input})
        intent = intent_result.content.strip().upper() if hasattr(intent_result, "content") else str(intent_result).strip().upper()

        if "CHITCHAT" in intent:
            prompt_chitchat = ChatPromptTemplate.from_messages([
                ("system", "Bạn là một trợ lý AI thân thiện. Hãy trả lời câu giao tiếp của người dùng một cách tự nhiên bằng Tiếng Việt."),
                ("human", "{input}"),
            ])
            chain_chitchat = prompt_chitchat | self.llm
            result = chain_chitchat.invoke({"input": user_input})
            return result.content if hasattr(result, "content") else result, None

        # Conservation RAG
        contextualize_q_system_prompt = (
            "Dựa trên lịch sử trò chuyện và câu hỏi mới nhất của người dùng, "
            "có thể câu hỏi mới đang tham chiếu đến ngữ cảnh trong lịch sử trò chuyện. "
            "Hãy viết lại câu hỏi thành một câu hỏi độc lập (standalone question) có thể tự hiểu được "
            "mà không cần lịch sử trò chuyện. "
            "KHÔNG trả lời câu hỏi, chỉ viết lại nếu cần thiết, nếu không thì trả về nguyên bản."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        history_aware_retriever = create_history_aware_retriever(
            self.llm, retriever, contextualize_q_prompt
        )
        qa_system_prompt = (
            "Bạn là một chuyên gia phân tích tài liệu. Nhiệm vụ của bạn là tổng hợp và trả lời dựa trên các ĐOẠN NGỮ CẢNH (Context) dưới đây.\n\n"
            "QUY TẮC QUAN TRỌNG:\n"
            "- Trả lời chi tiết, mạch lạc và bám sát vào ngữ cảnh.\n"
            "- Nếu người dùng hỏi câu hỏi tóm tắt chung chung (VD: 'Tài liệu này nói về gì?', 'Tóm tắt nội dung'), hãy tổng hợp ý chính từ các đoạn ngữ cảnh. Nếu có nhiều tài liệu, hãy chỉ rõ tài liệu nào nói về cái gì.\n"
            "- Tuyệt đối không bịa đặt. Nếu thông tin trong ngữ cảnh không đủ để trả lời trọn vẹn, hãy nói: 'Dựa trên các đoạn trích xuất hiện tại, tôi nhận thấy các thông tin sau: [Tóm tắt những gì có]...' thay vì từ chối.\n\n"
            "NGỮ CẢNH:\n{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        document_prompt = PromptTemplate.from_template(
            "Nguồn: {file_name}\nNội dung: {page_content}"
        )
        question_answer_chain = create_stuff_documents_chain(
            llm=self.llm, 
            prompt=qa_prompt,
            document_prompt=document_prompt
        )
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # 1. GỌI HÀM TRUY XUẤT CHUẨN CỦA LANGCHAIN
        # Hàm invoke() này chạy an toàn cho TẤT CẢ các loại Retriever (Vector, BM25, Ensemble)
        raw_docs = history_aware_retriever.invoke({"input": user_input, "chat_history": formatted_history})

        # 2. LỌC TÀI LIỆU (POST-FILTERING)
        relevant_docs = []
        if filter_filename and filter_filename != "Tất cả":
            for doc in raw_docs:
                doc_full_name = doc.metadata.get("file_name", "")
                doc_stem = Path(doc_full_name).stem if doc_full_name else ""
                if doc_stem == filter_filename or doc_full_name == filter_filename:
                    relevant_docs.append(doc)
        else:
            relevant_docs = raw_docs
        relevant_docs = relevant_docs[:6]

        if not relevant_docs:
            return "Xin lỗi, tôi không tìm thấy thông tin nào liên quan đến câu hỏi của bạn trong (các) tài liệu đã được chọn lọc.", []

        result = question_answer_chain.invoke({
            "input": user_input,
            "chat_history": formatted_history,
            "context": relevant_docs
        })

        answer = result if isinstance(result, str) else result.get("answer", result.get("output", str(result)))
        answer = answer.strip()

        return answer, relevant_docs

    def load_persistent_data(self):
        if os.path.exists(SAVE_DIR):
            # 1. Nạp danh sách file (JSON)
            if os.path.exists(META_FILE):
                with open(META_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    st.session_state["uploaded_filenames"] = data.get("uploaded_filenames", [])
                    st.session_state["uploaded_filenames_graph"] = data.get("uploaded_filenames_graph", [])
                    st.session_state["last_vector_key"] = data.get("last_vector_key", None)
                    st.session_state["last_graph_key"] = data.get("last_graph_key", None)

            # 2. Nạp Vector Store (FAISS)
            if os.path.exists(f"{SAVE_DIR}/index.faiss"):
                try:
                    vector_store = FAISS.load_local(
                        SAVE_DIR, 
                        self.embedder,
                        allow_dangerous_deserialization=True 
                    )
                    st.session_state.vector_store = vector_store
                    
                    # 3. Tái tạo lại cánh tay Retriever (Ensemble RAG)
                    all_docs = list(vector_store.docstore._dict.values())
                    bm25_retriever = BM25Retriever.from_documents(all_docs)
                    bm25_retriever.k = 3
                    faiss_retriever = vector_store.as_retriever(search_kwargs={"k": 3})
                    
                    st.session_state.retriever = EnsembleRetriever(
                        retrievers=[bm25_retriever, faiss_retriever], 
                        weights=[0.5, 0.5]
                    )
                except Exception as e:
                    print(f"Không thể nạp FAISS, có thể do file lỗi: {e}")