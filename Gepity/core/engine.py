from unittest import result

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama, OllamaLLM
from numpy import extract
from regex import D
from streamlit import status
from sympy import N, im
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
    def __init__(self, extract_model_name="qwen2.5:3b", response_model_name="qwen2.5:3b"):
        self.extract_llm = ChatOllama(
            model=extract_model_name, 
            temperature=0,
            num_ctx=4096,
            format="json",
        )

        # separate llm for response generation for better response quality
        self.response_llm = ChatOllama(
            model=response_model_name, 
            temperature=0.7, # higher temperature for more creative response generation
            num_ctx=4096,
        )

        self.graph = get_graph_connection()

        # Configure transformer
        self.transformer = LLMGraphTransformer(
            llm=self.extract_llm,
            # remove allowed_nodes and allowed_relationships to let llm extract any entity and relationship it finds in the text, which is more flexible and suitable for general documents
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
    
    def get_response(self, user_input):
        response = None
        if not self.graph:
            st.error("Không thể kết nối đến Neo4j")
            # make llm saif they can't access the graph instead of returning nothing
            response = self.response_llm.invoke(self.build_response_prompt(user_input))
            return response.content
        
        #extract entity relevant to user input from graph
        extract_prompt = self.build_extract_prompt(user_input)
        extraction_result = self.extract_llm.invoke(extract_prompt)
        entities = extraction_result.get("entities", [])

        if not entities:
            response = self.response_llm.invoke(self.build_response_prompt(user_input))
            return response.content
        
        # build graph context based on extracted entity
        graph_context_list = []
        for entity in entities:
            if len(entity) < 2:
                continue

            cypher_query = """
            MATCH (n)-[r]->(m)
            WHERE n.id CONTAINS $entity OR m.id CONTAINS $entity
            RETURN n.id + ' --[' + type(r) + ']--> ' + m.id AS relation
            LIMIT 15
            """
            result = self.graph.query(cypher_query, parameters={"entity": entity})
            graph_context_list.extend([record["relation"] for record in result])

        # join all relationships into a single string as graph context for response generation
        graph_context = "\n".join(list(set(graph_context_list)))

        if not graph_context:
            response = self.response_llm.invoke(self.build_response_prompt(user_input, entities=entities))
            return response.content
        
        # build prompt with graph context and user input
        prompt = self.build_response_prompt(user_input, entities=entities, context=graph_context)
        response = self.response_llm.invoke(prompt)

        return response.content, graph_context

        
    def build_response_prompt(self, user_input, entities = None, context=None):
        if not self.graph:
            if is_vietnamese(user_input):
                return f"""Xin lỗi, tôi không thể truy cập vào đồ thị kiến thức vào lúc này. Vui lòng thử lại sau."""
            else:
                return f"""Sorry, I cannot access the knowledge graph at the moment. Please try again later."""
            
        if not entities:
            if is_vietnamese(user_input):
                return f"""Xin lỗi, tôi không thể tìm thấy thực thể nào liên quan đến câu hỏi của bạn trong đồ thị kiến thức. Vui lòng thử lại với câu hỏi khác hoặc kiểm tra lại thông tin đã được cung cấp."""
            else:
                return f"""Sorry, I couldn't find any entities relevant to your question in the knowledge graph. Please try again with a different question or check the information provided."""

        if not context:
            # let llm said they can't find the context
            if is_vietnamese(user_input):
                return f"""Xin lỗi, tôi không thể tìm thấy ngữ cảnh liên quan đến câu hỏi của bạn trong đồ thị kiến thức. Vui lòng thử lại với câu hỏi khác hoặc kiểm tra lại thông tin đã được cung cấp."""
            else:
                return f"""Sorry, I couldn't find relevant context for your question in the knowledge graph. Please try again with a different question or check the information provided."""

        # build prompt with graph context and user input
        if is_vietnamese(user_input):
            return f"""Dựa trên input của người dùng, hãy sử dụng các mối quan hệ thực thể dưới đây (được trích xuất từ Knowledge Graph) để trả lời câu hỏi. 
            Nếu thông tin dưới đây không có câu trả lời, hãy dựa vào kiến thức của bạn nhưng ưu tiên dữ liệu từ Graph.

            Các mối quan hệ thực thể:
            {context}

            Câu hỏi: {user_input}
            
            Trả lời BẮT BUỘC bằng tiếng Việt (ngắn gọn, tập trung vào mối quan hệ giữa các thực thể):"""
        else: 
            return f"""Based on user input, use the entity relationships listed below (extracted from a knowledege graph) to answer the question.
            If the provided information does not have an answer, please use your availabled knowledge but remember to prioritize the graph's data.
            
            Entity relationships:
            {context}
            
            Question: {user_input}

            Answer(concise, focus on the relationships between entities):"""
 

    def build_extract_prompt(self, user_input):
        if is_vietnamese(user_input):
            return f"""Dựa trên input của người dùng, hãy trích xuất các thực thể và mối quan hệ liên quan từ đồ thị kiến thức. 
            Trả lời dưới dạng JSON với 2 trường: "entities" (danh sách các thực thể) và "relationships" (danh sách các mối quan hệ).
            Nếu không tìm thấy thực thể hoặc mối quan hệ nào liên quan, trả về {"entities": [], "relationships": []}.
            
            Input của người dùng: {user_input}
            
            JSON trả về:"""
        else:
            return f"""Based on the user input, extract relevant entities and relationships from the knowledge graph. 
            Respond in JSON format with 2 fields: "entities" (list of entities) and "relationships" (list of relationships).
            If no relevant entities or relationships are found, return {"entities": [], "relationships": []}.
            
            User input: {user_input}
            
            JSON response:"""
    
