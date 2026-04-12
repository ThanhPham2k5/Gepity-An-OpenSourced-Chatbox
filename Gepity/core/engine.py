from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from .processor import get_docs_from_uploaded_files, split_docs_into_chunks
from langchain_community.vectorstores import FAISS
from utils import is_vietnamese


class RAG_engine:
    def __init__(self, model_name="qwen2.5:7b"):
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
    
