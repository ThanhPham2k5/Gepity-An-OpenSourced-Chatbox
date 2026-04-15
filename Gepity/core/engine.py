from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os

from gliner import GLiNER
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama, OllamaLLM
from .processor import get_docs_from_uploaded_files, split_docs_into_chunks, extract_and_link_entities
from langchain_community.vectorstores import FAISS
from utils import is_vietnamese, setup_constraints
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
from langchain_experimental.graph_transformers import LLMGraphTransformer
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx
from dotenv import load_dotenv
load_dotenv()

def _is_running_in_streamlit():
    return get_script_run_ctx() is not None

class RAG_engine:
    def __init__(self, model_name="qwen2.5:3b"):
        self.llm = OllamaLLM(model=model_name, base_url="http://localhost:11434")
        # This tells the library where to save the model on your disk
        os.environ['HF_HOME'] = '/home/thien/OSSD/models_cache'
        self.embedder = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            model_kwargs={"device": "cpu", "token": os.getenv("HF_TOKEN")},
            encode_kwargs={"normalize_embeddings": True},
            cache_folder="/home/thien/OSSD/models_cache"
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
                return chain_chitchat.invoke({"input": user_input, "chat_history": formatted_history}).content, None
            else:
                return self.llm.invoke(user_input).content, None

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
        # Thay vì ép DB lọc (dễ gây lỗi), ta lấy kết quả ra rồi tự dùng Python để lọc
        relevant_docs = []
        if filter_filename and filter_filename != "Tất cả":
            for doc in raw_docs:
                if doc.metadata.get("file_name") == filter_filename:
                    relevant_docs.append(doc)
        else:
            relevant_docs = raw_docs

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
    def __init__(self, summary_model_name="qwen2.5:3b", response_model_name="qwen2.5:3b"):
        
        self.response_llm = ChatOllama(
            model=response_model_name, 
            temperature=0.7, # higher temperature for more creative response generation
            num_ctx=4096,
        )

        self.summary_llm = ChatOllama(
            model=summary_model_name,
            temperature=0,
            num_ctx=4096,
            format="json",
        )

        self.graph = get_graph_connection()
        setup_constraints(self.graph)

        # 1. Prompt cho Level 0 (Gom Entities)
        leaf_prompt = ChatPromptTemplate.from_messages([
            ("system", """Bạn là chuyên gia phân tích đồ thị tri thức. Dựa trên danh sách các thực thể (Entity) và mô tả của chúng, hãy tóm tắt cụm này.
            - 'title': Tên chủ đề chung của các thực thể.
            - 'summary': 1-2 câu mô tả mối liên hệ giữa các thực thể này.
            - 'full_content': Đoạn văn phân tích chi tiết vai trò của các thực thể và lý do chúng được xếp chung vào một cụm.
            - Output: BẮT BUỘC là JSON Object bằng Tiếng Việt: {{"title": "", "summary": "", "full_content": ""}} 
            Tuyệt đối không dùng markdown block."""),
            ("human", "Danh sách thực thể:\n{community_data}")
        ])

        # 2. Prompt cho Level 1+ (Gom Sub-communities)
        parent_prompt = ChatPromptTemplate.from_messages([
            ("system", """Bạn là chuyên gia tổng hợp thông tin vĩ mô. Dựa trên thông tin của các cụm chủ đề con (Sub-communities), hãy tổng hợp chúng thành một cụm chủ đề cha (Parent Community).
            - 'title': Tên chủ đề vĩ mô bao trùm tất cả các cụm con.
            - 'summary': 1-2 câu tóm tắt điểm giao thoa lớn nhất giữa các cụm con.
            - 'full_content': Đoạn văn phân tích bức tranh toàn cảnh, cách các cụm con này ghép lại để tạo thành một chủ đề lớn hơn.
            - Output: BẮT BUỘC là JSON Object bằng Tiếng Việt: {{"title": "", "summary": "", "full_content": ""}}
            Tuyệt đối không dùng markdown block."""),
            ("human", "Thông tin các cụm con:\n{community_data}")
        ])

        self.leaf_chain = leaf_prompt | self.summary_llm
        self.parent_chain = parent_prompt | self.summary_llm

        os.environ['HF_HOME'] = '/home/thien/OSSD/models_cache'
        self.embedder = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            model_kwargs={"device": "cpu", "token": os.getenv("HF_TOKEN")},
            encode_kwargs={"normalize_embeddings": True},
            cache_folder="/home/thien/OSSD/models_cache"
        )

        self.vector_index = get_vector_from_database(embedder=self.embedder)

        self.gliner_model = GLiNER.from_pretrained("urchade/gliner_multi")
        

    def process_document(self, uploaded_files, chunk_size, chunk_overlap):
        # read files and extract text
        all_docs = get_docs_from_uploaded_files(uploaded_files)

        # split documents into chunks
        all_chunks = split_docs_into_chunks(all_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap) 

        return all_chunks

    def sync_to_graph(self, all_chunks):
        in_streamlit = _is_running_in_streamlit()

        if not self.graph:
            if in_streamlit:
                st.error("Không thể kết nối đến Neo4j sau nhiều lần thử. Vui lòng kiểm tra console.")
            else:
                print("Không thể kết nối đến Neo4j sau nhiều lần thử. Vui lòng kiểm tra console.")
            return 0
        
        total_chunks = len(all_chunks)
        completed_count = 0

        if in_streamlit:
            progress_bar = st.progress(0)
            st.info(f"Đang xử lý song song {total_chunks} chunks...")
            status_text = st.empty()
        else:
            print(f"Đang xử lý song song {total_chunks} chunks...")

        # Use ThreadPoolExecutor with as_completed for real-time tracking
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Submit all tasks
            future_to_chunk = {executor.submit(self.process_single_chunk, chunk): chunk for chunk in all_chunks}
            
            for future in as_completed(future_to_chunk):
                completed_count += 1
                progress_pct = completed_count / total_chunks
                
                # Update UI
                if in_streamlit:
                    status_text.text(f"Tiến độ: {completed_count}/{total_chunks} chunks hoàn tất")
                    progress_bar.progress(progress_pct)
                else:
                    # Update console
                    if completed_count % 10 == 0:
                        print(f"Progress: [{completed_count}/{total_chunks}] - {progress_pct:.1%}", flush=True)

                # Optional: Catch errors from the thread
                try:
                    future.result() 
                except Exception as e:
                    print(f"Error in a thread: {e}")

        if in_streamlit:
            st.success("Đã hoàn thành xử lý toàn bộ chunks!")
        else:
            print("Hoàn tất quá trình đồng bộ.")

        # create vector index for graph
        self.create_vector_indexes()
        self.build_community()

        if in_streamlit:
            status_text.text("Đã xây dựng xong Lexical Knowledge Graph!")
        else:
            print("Knowledge Graph sync complete!")
    
    def create_vector_indexes(self):
        """Creates native vector indexes in Neo4j for Chunks and Entities"""
        # 768 is the standard dimension for paraphrase-multilingual-mpnet-base-v2
        cypher_queries = [
            """
            CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
            FOR (c:Chunk) ON (c.embedding)
            OPTIONS {indexConfig: {
              `vector.dimensions`: 768,
              `vector.similarity_function`: 'cosine'
            }}
            """,
            """
            CREATE VECTOR INDEX entity_embeddings IF NOT EXISTS
            FOR (e:Entity) ON (e.embedding)
            OPTIONS {indexConfig: {
              `vector.dimensions`: 768,
              `vector.similarity_function`: 'cosine'
            }}
            """
        ]
        for query in cypher_queries:
            try:
                self.graph.query(query)
            except Exception as e:
                print(f"Index creation note: {e}")

    def build_community(self):
        """Phát hiện và tóm tắt cộng đồng theo cấu trúc Hierarchical Lexical Graph"""

        in_streamlit = _is_running_in_streamlit()
        if not self.graph:
            return
        
        if in_streamlit:
            st.info("Bắt đầu xây dựng Communities...")
        else:
            print("Bắt đầu xây dựng Communities...")

        project_name = "knowledge_graph_projection"

        # 1. Dọn dẹp projection cũ nếu có
        self.graph.query(f"CALL gds.graph.drop('{project_name}', false) YIELD graphName")

        # 2. Tạo In-Memory Graph (Chỉ lấy Entity và quan hệ RELATES_TO)
        project_cypher = f"""
        CALL gds.graph.project(
            '{project_name}',
            'Entity',
            {{
                RELATES_TO: {{
                    orientation: 'UNDIRECTED'
                }}
            }}
        )
        """
        self.graph.query(project_cypher)

        # 3. Chạy Louvain và Stream kết quả thẳng về Python
        # Hàm này tự động gom nhóm các Entity có cùng communityId thành các mảng
        louvain_cypher = f"""
        CALL gds.louvain.stream('{project_name}')
        YIELD nodeId, communityId
        WITH gds.util.asNode(nodeId) AS e, communityId
        RETURN communityId, 
               collect({{entity_name: e.name, entity_desc: e.description}}) AS cluster_entities
        ORDER BY size(cluster_entities) DESC
        """
        
        results = self.graph.query(louvain_cypher)

        # 4. Xóa In-Memory Graph để giải phóng RAM
        self.graph.query(f"CALL gds.graph.drop('{project_name}', false) YIELD graphName")

        if not results:
            if in_streamlit:
                st.warning("Không tìm thấy cộng đồng nào. Vui lòng kiểm tra lại dữ liệu Entity.")
            else:
                print("Không tìm thấy cộng đồng nào. Vui lòng kiểm tra lại dữ liệu Entity.")
            return

        # 5. Xử lý và tạo Community Nodes (Level 0)
        if in_streamlit:
            progress_bar = st.progress(0)
        level_0_community_ids = []

        # Trong GDS, mỗi record đã là một cụm (community) hoàn chỉnh, không cần chia batch ảo nữa.
        for i, record in enumerate(results):
            community_group_id = record['communityId']
            cluster = record['cluster_entities']
            
            # Bỏ qua các cụm quá nhỏ
            if len(cluster) < 5:
                continue
            
            weight = len(cluster)
            level = 0
            community_id = f"COMM_L{level}_{community_group_id}"
            level_0_community_ids.append(community_id)

            # Chuẩn bị dữ liệu cho LLM
            context_str = "\n".join([f"- {c['entity_name']}: {c['entity_desc']}" for c in cluster])

            try:
                # LLM tạo Title, Summary, Full Content
                response = self.leaf_chain.invoke({"community_data": context_str})
                community_info = json.loads(response.content)
                
                title = community_info.get("title", f"Community {community_group_id}")
                summary = community_info.get("summary", "Không có tóm tắt")
                full_content = community_info.get("full_content", "Không có nội dung")

                # Lưu vào Neo4j
                save_cypher = """
                MERGE (c:Community {id: $community_id})
                SET c.title = $title,
                    c.level = $level,
                    c.summary = $summary,
                    c.full_content = $full_content,
                    c.weight = $weight
                
                WITH c
                UNWIND $entity_names AS e_name
                MATCH (e:Entity {name: e_name})
                MERGE (e)-[:IN_COMMUNITY]->(c)
                """
                
                entity_names = [c['entity_name'] for c in cluster]
                
                self.graph.query(save_cypher, {
                    "community_id": community_id,
                    "title": title,
                    "level": level,
                    "summary": summary,
                    "full_content": full_content,
                    "weight": weight,
                    "entity_names": entity_names
                })

            except Exception as e:
                print(f"Lỗi khi tạo community {community_id}: {e}")
            
            if in_streamlit:
                progress_bar.progress((i + 1) / len(results))

        # Gọi hàm tạo Level 1 như cũ
        if in_streamlit:
            st.info("Đang xây dựng Parent Communities (Level 1)...")
        else:
            print("Đang xây dựng Parent Communities (Level 1)...")
        if len(level_0_community_ids) > 1:
            self._build_parent_communities(level_0_community_ids)

        if in_streamlit:
            st.success("Hoàn tất xây dựng cấu trúc Community GraphRAG!")
        else:
            print("Hoàn tất xây dựng cấu trúc Community GraphRAG!")


    def _build_parent_communities(self, child_community_ids, current_level=1):
        """
        Xây dựng Parent Communities theo từng batch để tránh quá tải Context Window của LLM.
        Nếu tạo ra nhiều hơn 1 Parent Community, tự động đệ quy lên Level tiếp theo.
        """
        in_streamlit = _is_running_in_streamlit()

        if in_streamlit:
            st.info(f"Đang tổng hợp Level {current_level} Communities...")
        else:
            print(f"Đang tổng hợp Level {current_level} Communities...")

        # Lấy summary của các child communities từ Neo4j
        fetch_cypher = """
        MATCH (c:Community) 
        WHERE c.id IN $ids 
        RETURN c.id AS id, c.title AS title, c.summary AS summary
        """
        children = self.graph.query(fetch_cypher, {"ids": child_community_ids})
        
        if not children:
            return

        # Nhóm các child communities thành các batch
        batch_size = 5 
        parent_community_ids = []

        if in_streamlit:
            progress_bar = st.progress(0)

        for i in range(0, len(children), batch_size):
            batch = children[i : i + batch_size]
            
            parent_id = f"COMM_L{current_level}_{i}"
            parent_community_ids.append(parent_id)
            weight = len(batch)
            
            # Chuẩn bị context từ các cụm con
            context_str = "\n".join([f"- Cụm {c['title']}: {c['summary']}" for c in batch])
            
            try:
                # Gọi LLM để tổng hợp
                response = self.parent_chain.invoke({"community_data": context_str})
                parent_info = json.loads(response.content)
                
                # Lưu Parent Community và tạo link PARENT_COMMUNITY xuống các cụm con
                parent_cypher = """
                MERGE (p:Community {id: $parent_id})
                SET p.title = $title,
                    p.level = $level,
                    p.summary = $summary,
                    p.full_content = $full_content,
                    p.weight = $weight
                
                WITH p
                UNWIND $child_ids AS c_id
                MATCH (c:Community {id: c_id})
                MERGE (c)-[:PARENT_COMMUNITY]->(p)
                """
                
                batch_ids = [c['id'] for c in batch]
                self.graph.query(parent_cypher, {
                    "parent_id": parent_id,
                    "title": parent_info.get("title", f"Tổng hợp Level {current_level}"),
                    "level": current_level,
                    "summary": parent_info.get("summary", ""),
                    "full_content": parent_info.get("full_content", ""),
                    "weight": weight,
                    "child_ids": batch_ids
                })

            except Exception as e:
                print(f"Lỗi khi tạo parent community {parent_id}: {e}")
            
            if in_streamlit:
                progress_bar.progress(min((i + batch_size) / len(children), 1.0))

        # --- ĐỆ QUY (RECURSION) ---
        # Nếu chúng ta tạo ra nhiều hơn 1 Parent Community ở level này, 
        # tiếp tục gom chúng lại thành Level cao hơn cho đến khi chỉ còn 1 Master Node.
        if len(parent_community_ids) > 1:
            self._build_parent_communities(parent_community_ids, current_level=current_level + 1)
        else:
            if in_streamlit:
                st.success(f"Đã đạt đến Root Community ở Level {current_level}!")
            else:
                print(f"Đã đạt đến Root Community ở Level {current_level}!")
    
    def get_response(self, user_input):

        if not self.graph:
            st.error("Không thể kết nối đến Neo4j")
            if is_vietnamese(user_input):
                return f"""Xin lỗi, tôi không thể truy cập vào đồ thị kiến thức vào lúc này. Vui lòng thử lại sau."""
            else:
                return f"""Sorry, I cannot access the knowledge graph at the moment. Please try again later."""
        
        #extract nodes relevant to user input from graph
        relevant_nodes = self.vector_index.similarity_search(user_input, k=5)
        
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
        
    def process_single_chunk(self, chunk):
        # Prepare Document and Chunk Data
        doc_name = chunk.metadata.get("filename", "unknown")
        doc_source = chunk.metadata.get("source", "unknown")
        chunk_text = chunk.page_content
        chunk_id = hashlib.md5(chunk_text.encode('utf-8')).hexdigest()

        # generate embeddings for the chunk
        chunk_embedding = self.embedder.embed_query(chunk_text)

        # extract entities and relationships via LLM
        try:
            # get entities and relationships
            entities, relationships = extract_and_link_entities(chunk_text=chunk_text, gliner_model=self.gliner_model)

            # process and embeds entities
            batch_entities = []
            for ent in entities:
                # Generate embedding for the entity
                ent_embedding = self.embedder.embed_query(f"{ent['name']} ({ent['label']})")
                # add entity to batch
                batch_entities.append({
                    "name": ent['name'],
                    "desc": ent['description'],
                    "label": ent['label'],
                    "embedding": ent_embedding
                })

            # single batch query
            with self.graph._driver.session(database="neo4j") as session:
                def write_transaction(tx):
                    sync_query = """
                    // Create Document and Chunk
                    MERGE (d:Document {source: $doc_source}) SET d.name = $doc_name
                    MERGE (c:Chunk {id: $chunk_id})
                    SET c.text = $chunk_text, c.embedding = $chunk_embedding
                    MERGE (c)-[:PART_OF]->(d)

                    // Batch Entities
                    WITH c
                    UNWIND $entities AS ent
                    MERGE (e:Entity {name: ent.name})
                    SET e.description = ent.desc, e.embedding = ent.embedding, e.label = ent.label
                    MERGE (c)-[:HAS_ENTITY]->(e)

                    // Batch Relationships (Co Occurence)
                    WITH c
                    UNWIND $rels AS rel
                    MATCH (src:Entity {name: rel.source})
                    MATCH (tgt:Entity {name: rel.target})
                    MERGE (src)-[r:RELATES_TO]->(tgt)
                    ON CREATE SET r.weight = 1, r.type = rel.type
                    ON MATCH SET r.weight = r.weight + 1
                    """

                    tx.run(sync_query, {
                        "doc_source": doc_source,
                        "doc_name": doc_name,
                        "chunk_id": chunk_id,
                        "chunk_text": chunk_text,
                        "chunk_embedding": chunk_embedding,
                        "entities": batch_entities,
                        "rels": relationships
                    })

                # If write_transaction fails, the session.execute_write will rollback automatically
                session.execute_write(write_transaction)
        except Exception as e:
            print(f"Lỗi chunk (Đã rollback): {e}", flush=True)
    
