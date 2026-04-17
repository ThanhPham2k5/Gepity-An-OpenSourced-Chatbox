import os
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
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.ai import AIMessage
from dotenv import load_dotenv
load_dotenv()

# Use for Windows IP
# WINDOWS_IP = "172.25.64.1"

# Use for WSL IP
WINDOWS_IP = "localhost"
MODELS_CACHE = "../../models_cache"

class RAG_engine:
    def __init__(self, model_name="qwen2.5:3b"):
        self.llm = OllamaLLM(model=model_name, base_url=f"http://{WINDOWS_IP}:11434")
        # This tells the library where to save the model on your disk
        os.environ['HF_HOME'] = '../../models_cache'
        self.embedder = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            model_kwargs={"device": "cpu", "token": os.getenv("HF_TOKEN")},
            encode_kwargs={"normalize_embeddings": True},
            cache_folder="../../models_cache"
        )

    def process_document(self, uploaded_files, chunk_size=500, chunk_overlap=50):
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

        vector_store = FAISS.from_documents(all_chunks, self.embedder)
        faiss_retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        bm25_retriever = BM25Retriever.from_documents(all_chunks)
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
            "Bạn là một bộ phân loại ý định. Nhiệm vụ của bạn là phân tích câu hỏi "
            "hoặc lời nói của người dùng và quyết định xem nó thuộc loại nào trong 2 loại sau:\n"
            "1. 'CHITCHAT': Những câu giao tiếp xã giao thông thường hoặc những câu hỏi không liên quan đến tài liệu, lịch sử trò chuyện (VD: Xin chào, bạn khỏe không, cảm ơn, tạm biệt) "
            "hoặc những câu hỏi chung chung không cần tra cứu tài liệu.\n"
            "2. 'DOC_SEARCH': Những câu hỏi cần tra cứu thông tin cụ thể, phân tích, hoặc truy vấn kiến thức có liên quan đến tài liệu, lịch sử trò chuyện.\n\n"
            "Chỉ trả về ĐÚNG 1 TỪ duy nhất: 'CHITCHAT' hoặc 'DOC_SEARCH'."
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
                ("system", "Bạn là một trợ lý AI thân thiện. Hãy trả lời câu giao tiếp của người dùng một cách tự nhiên."),
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
            "Bạn là trợ lý giải đáp thắc mắc. Chỉ sử dụng tài liệu được cung cấp để trả lời.\n"
            "QUY TẮC:\n"
            "- Trả lời ngắn gọn, trực tiếp vào câu hỏi.\n"
            "- KHÔNG giải thích thêm, KHÔNG chào hỏi, KHÔNG gợi ý thêm.\n"
            "- Nếu không có trong Context, chỉ trả lời: 'Thông tin này không có trong tài liệu.'\n\n"
            "CONTEXT:\n{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # 1. GỌI HÀM TRUY XUẤT CHUẨN CỦA LANGCHAIN
        # Hàm invoke() này chạy an toàn cho TẤT CẢ các loại Retriever (Vector, BM25, Ensemble)
        raw_docs = history_aware_retriever.invoke({"input": user_input, "chat_history": formatted_history})

        # 2. LỌC TÀI LIỆU (POST-FILTERING)
        relevant_docs = []
        if filter_filename and filter_filename != "Tất cả":
            for doc in raw_docs:
                if doc.metadata.get("file_name") == filter_filename:
                    relevant_docs.append(doc)
        else:
            relevant_docs = raw_docs
        relevant_docs = relevant_docs[:4]

        result = question_answer_chain.invoke({
            "input": user_input,
            "chat_history": formatted_history,
            "context": relevant_docs
        })

        answer = result if isinstance(result, str) else result.get("output", str(result))

        return answer, relevant_docs
    
    def build_prompt(self, context: str, user_input: str) -> str:
        if is_vietnamese(user_input): # if user input is in Vietnamese, response in Vietnamese
            return f""" Sử dụng ngữ cảnh sau đây để trả lời câu hỏi.
            Nếu bạn không biết, chỉ cần nói là bạn không biết.
            Trả lời ngắn gọn (3-4 câu) BẮT BUỘC bằng tiếng Việt.
            
            Ngữ cảnh: {context}
            
            Câu hỏi: {user_input}
            
            Trả lời:"""
        else: # default to English
            return f"""Use the following context to answer the question.
            If you don't know the answer, just say you don't know.
            Keep answer concise (3-4 sentences).
            
            Context: {context}
            
            Question: {user_input}
            
            Answer:"""
