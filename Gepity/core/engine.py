from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama, OllamaLLM
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
from database import get_graph_connection, get_vector_from_database
# from langchain_experimental.graph_transformers import LLMGraphTransformer
import streamlit as st

class RAG_engine:
    def __init__(self, model_name="qwen2.5:7b"):
        self.llm = OllamaLLM(model=model_name, base_url="http://localhost:11434")
        self.embedder = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
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
                    formatted_history.append(AIMessage(content=msg["content"]))

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
            "Bạn là trợ lý giải đáp thắc mắc. Sử dụng các tài liệu được cung cấp dưới đây để trả lời câu hỏi.\n\n"
            "{context}"
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

        response = question_answer_chain.invoke({
            "input": user_input,
            "chat_history": formatted_history,
            "context": relevant_docs
        })

        return response, relevant_docs
    
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


        graph_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """Bạn là một chuyên gia về Knowledge Graph. Nhiệm vụ của bạn là trích xuất các thực thể và mối quan hệ từ văn bản được cung cấp theo các nguyên tắc sau:

                1. THỰC THỂ (ENTITIES): 
                - Sử dụng danh từ cụ thể, ngắn gọn. 
                - Label của thực thể phải là một từ đơn mô tả loại (ví dụ: 'Technology', 'Concept', 'Component', 'Person').

                2. MỐI QUAN HỆ (RELATIONSHIPS):
                - Sử dụng động từ viết hoa, nối bằng dấu gạch dưới (ví dụ: 'SỬ_DỤNG', 'THUỘC_VỀ').

                3. TÍNH ĐỒNG NHẤT:
                - Quy về một tên gọi chuẩn nhất cho các thực thể trùng lặp.
                - Luôn trích xuất thuộc tính 'description' nếu có.

                4. ĐỊNH DẠNG: Trả về kết quả dưới dạng đồ thị Nodes và Edges."""
            ),
            (
                "human",
                "Hãy trích xuất thực thể và quan hệ từ đoạn văn bản sau: {input}"
            ),
        ])

        # Configure transformer
        self.transformer = LLMGraphTransformer(
            llm=self.extract_llm,
            node_properties=["description"],
            prompt= graph_prompt,
        )

        self.embedder = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        self.vector_node = get_vector_from_database(embedder=self.embedder)

    def process_document(self, uploaded_files):
        # read files and extract text
        all_docs = get_docs_from_uploaded_files(uploaded_files)

        # Bổ sung logic: Ghi đè hoặc thêm thông tin file_name vào metadata
        for doc in all_docs:
            # Lấy tên file từ file_path hoặc thuộc tính có sẵn
            actual_file_name = doc.metadata.get("source", "Unknown_File").split("/")[-1]
            doc.metadata["file_name"] = actual_file_name

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

                graph_docs = self.clean_graph_documents(graph_docs)
          
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

        # reload vector after adding new nodes
        self.vector_node = get_vector_from_database(embedder=self.embedder)
        status_text.text("Successfully built Knowledge Graph!")
        return successful_docs
    
    def get_response(self, user_input):

        if not self.graph:
            st.error("Không thể kết nối đến Neo4j")
            if is_vietnamese(user_input):
                return f"""Xin lỗi, tôi không thể truy cập vào đồ thị kiến thức vào lúc này. Vui lòng thử lại sau."""
            else:
                return f"""Sorry, I cannot access the knowledge graph at the moment. Please try again later."""
        
        #extract nodes relevant to user input from graph
        relevant_nodes = self.vector_node.similarity_search(user_input, k=5)
        
        # build graph context based on extracted entity
        graph_context_list = []
        entity_ids = [node.page_content for node in relevant_nodes]
        cypher_query = """
        MATCH (n)-[r]-(m)
        WHERE n.id IN $entity_ids 
           OR n.id CONTAINS $query_upper 
           OR n.id CONTAINS $query_title
        RETURN n.id AS source, type(r) AS rel, m.id AS target, n.description AS desc
        LIMIT 10
        """
        
        query_upper = user_input.upper()
        query_title = user_input.title()

        results = self.graph.query(cypher_query, {
            "entity_ids": entity_ids,
            "query_upper": query_upper,
            "query_title": query_title
        })

        for record in results:
            relation_str = f"{record['source']} --[{record['rel']}]--> {record['target']}"
            
            description = record.get('desc') 
            if description:
                relation_str += f" ({description})"
                
            graph_context_list.append(relation_str)

        # join all relationships into a single string as graph context for response generation
        graph_context = "\n".join(list(set(graph_context_list)))

        if not graph_context:
            if is_vietnamese(user_input):
                return f"""Xin lỗi, tôi không thể tìm thấy ngữ cảnh liên quan đến câu hỏi của bạn trong đồ thị kiến thức. Vui lòng thử lại với câu hỏi khác hoặc kiểm tra lại thông tin đã được cung cấp."""
            else:
                return f"""Sorry, I couldn't find relevant context for your question in the knowledge graph. Please try again with a different question or check the information provided."""
        
        # build prompt with graph context and user input
        print(f"DEBUG: Context length: {len(graph_context)} characters")
        prompt = self.build_response_prompt(user_input, context=graph_context)
        response = self.response_llm.invoke(prompt)

        return response.content
        
        
    def build_response_prompt(self, user_input, context):
        # build prompt with graph context and user input
        if is_vietnamese(user_input):
            return f"""Dựa trên input của người dùng, hãy sử dụng các mối quan hệ thực thể dưới đây (được trích xuất từ Knowledge Graph) để trả lời câu hỏi. 
            Nếu thông tin dưới đây không có câu trả lời, hãy dựa vào kiến thức của bạn nhưng hãy thông báo trước nếu bạn làm vậy.

            Các mối quan hệ thực thể:
            {context}

            Câu hỏi: {user_input}
            
            Trả lời BẮT BUỘC bằng tiếng Việt (ngắn gọn, tập trung vào mối quan hệ giữa các thực thể):"""
        else: 
            return f"""Based on user input, use the entity relationships listed below (extracted from a knowledege graph) to answer the question.
            If the provided information does not have an answer, then use your availabled knowledge but you must notify when you do.
            
            Entity relationships:
            {context}
            
            Question: {user_input}

            Answer(concise, focus on the relationships between entities):"""
        
    def clean_graph_documents(self, graph_docs):
        """Làm sạch sâu hơn để tăng tỷ lệ kết nối các node"""
        for doc in graph_docs:
            for node in doc.nodes:
                # Loại bỏ khoảng trắng thừa, viết hoa đầu từ, và xử lý viết tắt phổ biến
                clean_id = node.id.strip().title() 
                node.id = clean_id

            for rel in doc.relationships:
                rel.type = rel.type.strip().replace(" ", "_").upper()
                rel.source.id = rel.source.id.strip().title()
                rel.target.id = rel.target.id.strip().title()
        return graph_docs
 

    # def build_extract_prompt(self, user_input):
    #     if is_vietnamese(user_input):
    #         return f"""Dựa trên input của người dùng, hãy trích xuất các thực thể và mối quan hệ liên quan từ đồ thị kiến thức. 
    #         Trả lời dưới dạng JSON với 2 trường: "entities" (danh sách các thực thể) và "relationships" (danh sách các mối quan hệ).
    #         Nếu không tìm thấy thực thể hoặc mối quan hệ nào liên quan, trả về {{"entities": [], "relationships": []}}.
            
    #         Input của người dùng: {user_input}
            
    #         JSON trả về:"""
    #     else:
    #         return f"""Based on the user input, extract relevant entities and relationships from the knowledge graph. 
    #         Respond in JSON format with 2 fields: "entities" (list of entities) and "relationships" (list of relationships).
    #         If no relevant entities or relationships are found, return {{"entities": [], "relationships": []}}.
            
    #         User input: {user_input}
            
    #         JSON response:"""
    
