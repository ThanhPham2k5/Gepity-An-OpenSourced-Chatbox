from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama, OllamaLLM
from regex import D
from streamlit import status
from sympy import im
from .processor import get_docs_from_uploaded_files, split_docs_into_chunks
from langchain_community.vectorstores import FAISS
from utils import is_vietnamese
from database import get_graph_connection
from langchain_experimental.graph_transformers import LLMGraphTransformer
import streamlit as st


class RAG_engine:
    def __init__(self, model_name="qwen2.5:3b"):
        self.llm = OllamaLLM(model=model_name, base_url="http://localhost:11434")
        self.embedder = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

    def process_document(self, uploaded_files):
        # read files and extract text
        all_docs = get_docs_from_uploaded_files(uploaded_files)

        # split documents into chunks
        all_chunks = split_docs_into_chunks(all_docs)

        # store vector in session
        vector_store = FAISS.from_documents(all_chunks, self.embedder)

        # create retriever 
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}, # retrieve top 3 most relevant chunks
        )

        return retriever, vector_store, len(all_chunks), len(all_docs)

    def get_response(self, user_input, retriever):
        # if no document uploaded, llm will answer directly
        if retriever is None:
            return self.llm.invoke(user_input)

        # retrieve relevant documents based on user input
        relevant_docs = retriever.invoke(user_input)

        # Context building
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # build prompt with context and user input
        prompt = self.build_prompt(context, user_input)

        # llm response
        response = self.llm.invoke(prompt)

        return response
    
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

class Graph_engine:
    def __init__(self, model_name="qwen2.5:3b"):
        self.graph_extracting_llm = ChatOllama(
            model=model_name, 
            temperature=0,
            num_ctx=4096,
            format="json",
        )
        self.graph = get_graph_connection()

        # Configure transformer
        self.transformer = LLMGraphTransformer(
            llm=self.graph_extracting_llm,
            # allowed_nodes=["Concept", "Entity", "Technology", "Person", "Organization", "Location", "Event"],
            # allowed_relationships=["RELATED_TO", "USES", "PART_OF", "LOCATED_IN", "WORKS_FOR", "KNOWS", "ABOUT", "CAUSES", "HAS_PROPERTY"]
        )

    def process_document(self, uploaded_files):
        # read files and extract text
        all_docs = get_docs_from_uploaded_files(uploaded_files)

        # split documents into chunks
        all_chunks = split_docs_into_chunks(all_docs, chunk_size=800, chunk_overlap=100) 

        return all_chunks

    def sync_to_graph(self, all_chunks):
        if not self.graph:
            self.graph = get_graph_connection()
        
        if not self.graph:
            st.error("Không thể kết nối đến Neo4j sau nhiều lần thử. Vui lòng kiểm tra console.")
            return 0
        
        total_chunks = len(all_chunks)
        progress_bar = st.progress(0)
        status_text = st.empty()

        batch_size = 2 # chunks per batch, adjust based on performance
        successful_docs = 0
        for i in range(0, total_chunks, batch_size):
            batch = all_chunks[i : i + batch_size]
            status_text.text(f"Đang phân tích Graph: {i}/{total_chunks} đoạn văn...")
            try:
                #extract entity and relationship from batch, then sync to graph
                graph_docs = self.transformer.convert_to_graph_documents(batch)
          
                self.graph.add_graph_documents(
                    graph_docs, 
                    baseEntityLabel=True, 
                    include_source=True
                )
                successful_docs += len(batch)
            except Exception as e:
                # if error occurs, skip the batch and continue with next one
                st.warning(f"Bỏ qua batch {i} do lỗi xử lý: {e}")
            
            # Update progress bar
            progress_bar.progress(min((i + batch_size) / total_chunks, 1.0))

        status_text.text("Successfully built Knowledge Graph!")
        return successful_docs
    
