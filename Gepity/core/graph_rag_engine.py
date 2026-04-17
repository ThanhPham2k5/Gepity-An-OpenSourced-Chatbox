from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from gliner import GLiNER
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from .processor import get_docs_from_uploaded_files, split_docs_into_chunks, extract_and_link_entities
from .builder import build_local_response_prompt, build_global_reduce_prompt, build_router_prompt, build_leaf_prompt, build_parent_prompt, build_global_context_from_result, build_local_context_from_result
from utils import is_vietnamese, setup_constraints

from langchain_core.prompts.chat import ChatPromptTemplate
from database import get_graph_connection, get_vector_from_index
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx
from dotenv import load_dotenv
load_dotenv()

def _is_running_in_streamlit():
    return get_script_run_ctx() is not None

class Graph_engine:
    def __init__(self, summary_model_name="qwen2.5:3b", response_model_name="qwen2.5:3b"):
        
        # use to response 
        self.response_llm = ChatOllama(
            model=response_model_name, 
            temperature=0.7, # higher temperature for more creative response generation
            num_ctx=4096,
        )

        # use to determine what type of question the user is asking
        self.router_llm = ChatOllama(
            model="qwen2.5:3b",
            temperature=0,          
            format="json",            
            num_ctx=2048,             
            num_thread=4
        )

        # use to build community summary
        self.summary_llm = ChatOllama(
            model=summary_model_name,
            temperature=0,
            num_ctx=4096,
            format="json",
        )

        self.graph = get_graph_connection()
        setup_constraints(self.graph)

        # Prompt for router
        router_prompt = build_router_prompt()

        # Prompt for level 0 community
        leaf_prompt = build_leaf_prompt()

        # Prompt for level 1+ community
        parent_prompt = build_parent_prompt()

        self.router_chain = router_prompt | self.router_llm
        self.leaf_chain = leaf_prompt | self.summary_llm
        self.parent_chain = parent_prompt | self.summary_llm

        os.environ['HF_HOME'] = '../../models_cache'
        self.embedder = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            model_kwargs={"device": "cpu", "token": os.getenv("HF_TOKEN")},
            encode_kwargs={"normalize_embeddings": True},
            cache_folder="../../models_cache"
        )

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
            st.info(f"Đang xử lý {total_chunks} chunks...")
            status_text = st.empty()
        else:
            print(f"Đang xử lý {total_chunks} chunks...")

        # Use ThreadPoolExecutor with as_completed for real-time tracking
        with ThreadPoolExecutor(max_workers=10) as executor:
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
                        print(f"Tiến độ: [{completed_count}/{total_chunks}] - {progress_pct:.1%}", flush=True)

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
            status_text = st.empty()
            st.info("Bắt đầu xây dựng Communities...")
        else:
            print("Bắt đầu xây dựng Communities...")

        project_name = "knowledge_graph_projection"

        # Dọn dẹp projection cũ nếu có
        self.graph.query(f"CALL gds.graph.drop('{project_name}', false) YIELD graphName")

        # Tạo In-Memory Graph (Chỉ lấy Entity và quan hệ RELATES_TO)
        project_cypher = f"""
        CALL gds.graph.project(
            '{project_name}',
            'Entity',
            {{
                RELATES_TO: {{
                    orientation: 'UNDIRECTED',
                    properties: {{
                        weight: {{
                            property: 'weight',
                            defaultValue: 1.0
                        }}
                    }}
                }}
            }}
        )
        """
        self.graph.query(project_cypher)

        # Chạy leiden và Stream kết quả thẳng về Python
        leiden_cypher = f"""
        CALL gds.leiden.stream('{project_name}', {{
            relationshipWeightProperty: 'weight',
            includeIntermediateCommunities: false
        }})
        YIELD nodeId, communityId
        WITH gds.util.asNode(nodeId) AS e, communityId
        RETURN communityId, 
               collect({{entity_name: e.name, entity_desc: e.description}}) AS cluster_entities
        ORDER BY size(cluster_entities) DESC
        """
        results = self.graph.query(leiden_cypher)

        # Xóa In-Memory Graph để giải phóng RAM
        self.graph.query(f"CALL gds.graph.drop('{project_name}', false) YIELD graphName")

        if not results:
            if in_streamlit:
                st.warning("Không tìm thấy cộng đồng nào. Vui lòng kiểm tra lại dữ liệu Entity.")
            else:
                print("Không tìm thấy cộng đồng nào. Vui lòng kiểm tra lại dữ liệu Entity.")
            return

        # Xử lý và tạo Community Nodes (Level 0)  
        level_0_community_ids = []

        
        processed_communities = []

        # for record in results:
        #     result = self.process_community_worker(record, 0)
        #     if result:
        #         processed_communities.append(result)
        #         msg = f"Tiến độ: đã xử lý {len(processed_communities)} community"
        #         if in_streamlit:
        #             status_text.text(msg)
        #         else:
        #             print(msg)

        # parallelize llm call
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_complete = {executor.submit(self.process_community_worker, rec, 0): rec for rec in results}
            
            for future in as_completed(future_to_complete):
                res = future.result()
                if res:
                    processed_communities.append(res)
                    msg = f"Tiến độ: đã xử lý {len(processed_communities)} community"
                    if in_streamlit:
                        status_text.text(msg)
                    else:
                        print(msg, flush=True)

        # single batch query
        if processed_communities:
            save_batch_cypher = """
            UNWIND $data AS item
            MERGE (c:Community {id: item.community_id})
            SET c.title = item.title,
                c.level = item.level,
                c.summary = item.summary,
                c.full_content = item.full_content,
                c.weight = item.weight
            WITH c, item
            UNWIND item.entity_names AS e_name
            MATCH (e:Entity {name: e_name})
            MERGE (e)-[:IN_COMMUNITY]->(c)
            """
            self.graph.query(save_batch_cypher, {"data": processed_communities})
        level_0_community_ids = [c['community_id'] for c in processed_communities]

        # Gọi hàm tạo Level 1 như cũ
        if in_streamlit:
            st.info("Đang xây dựng Parent Communities (Level 1)...")
        else:
            print("Đang xây dựng Parent Communities (Level 1)...")
        if len(level_0_community_ids) > 1:
            self.build_parent_communities(level_0_community_ids)

        if in_streamlit:
            st.success("Hoàn tất xây dựng Community!")
        else:
            print("Hoàn tất xây dựng Community!")


    def build_parent_communities(self, child_community_ids, current_level=1):
        """
        Xây dựng Parent Communities bằng cách gom các cụm con và tóm tắt song song.
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
        batches = [children[i : i + batch_size] for i in range(0, len(children), batch_size)]
        
        parent_results = []

        # for i in range(0, len(children), batch_size):
        #     batch = children[i : i + batch_size]
        #     result = self.summarize_parent_batch(batch, current_level, i)
        #     if result:
        #         parent_results.append(result)

        # parallelize llm call
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_batch = {
                executor.submit(self.summarize_parent_batch, batch, current_level, i): i 
                for i, batch in enumerate(batches)
            }
            
            for future in as_completed(future_to_batch):
                res = future.result()
                if res:
                    parent_results.append(res)

        # single batch query
        if parent_results:
            save_parent_cypher = """
            UNWIND $data AS item
            MERGE (p:Community {id: item.parent_id})
            SET p.title = item.title,
                p.level = item.level,
                p.summary = item.summary,
                p.full_content = item.full_content,
                p.weight = item.weight
            WITH p, item
            UNWIND item.child_ids AS c_id
            MATCH (c:Community {id: c_id})
            MERGE (c)-[:PARENT_COMMUNITY]->(p)
            """
            self.graph.query(save_parent_cypher, {"data": parent_results})

        # Recursive: if there's more than one parent, continue to higher level
        new_parent_ids = [p['parent_id'] for p in parent_results]
        
        if len(new_parent_ids) > 1:
            self.build_parent_communities(new_parent_ids, current_level=current_level + 1)
        else:
            msg = f"Đã đạt đến Root Community ở Level {current_level}!"
            if in_streamlit:
                st.success(msg)
            else:
                print(msg)

    def global_search(self, level=None):
        if not level: # if level was not specified, get the second highest level
            level_query = """
            MATCH (c:Community)
            ORDER BY c.level DESC
            SKIP 1
            LIMIT 1
            RETURN c.level as level
            """
            level_data = self.graph.query(level_query)
            level = level_data[0]['level']

        retrieval_query = """
        MATCH (comm:Community)
        WHERE comm.level = $level AND comm.full_content <> ""

        RETURN comm.full_content
        """

        results = self.graph.query(retrieval_query, {
            "level": level
        })

        return results

    def local_search(self, user_input, threshold=0.7, top_k=5):
        # embed usesr input for vector search
        query_vector = self.embedder.embed_query(user_input)

        # use cypher to query relevant chunks, entities and community summaries for said entities
        retrieval_query = """
        CALL db.index.vector.queryNodes('chunk_embeddings', $top_k, $query_vector)
        YIELD node AS chunk, score
        WHERE score >= $threshold

        OPTIONAL MATCH (chunk)-[:HAS_ENTITY]->(e:Entity)

        OPTIONAL MATCH (e)-[:IN_COMMUNITY]->(comm:Community)
        
        RETURN
            chunk.text as text,
            chunk.page as page_number,
            score,
            collect(DISTINCT e.name) as related_entities,
            collect(DISTINCT comm.summary) as related_community_summaries
        """

        results = self.graph.query(retrieval_query, {
            "query_vector": query_vector,
            "top_k": top_k,
            "threshold": threshold
        })
        
        return results
        
    def get_response(self, user_input):
        if not self.graph:
            st.error("Không thể kết nối đến Neo4j")
            if is_vietnamese(user_input):
                return f"""Xin lỗi, tôi không thể truy cập vào đồ thị kiến thức vào lúc này. Vui lòng thử lại sau."""
            else:
                return f"""Sorry, I cannot access the knowledge graph at the moment. Please try again later."""
        
        # get router response
        router_response = self.router_chain.invoke(user_input)

        # clean up response
        router_content = router_response.content.replace("```json", "").replace("```", "").strip()
        router_info = json.loads(router_content)

        question_type = router_info.get("type", "general")

        if question_type == "global":
            raw_results = self.global_search(level=0)
            global_context = build_global_context_from_result(raw_results)

            if not global_context:
                if is_vietnamese(user_input):
                    return f"""Xin lỗi, tôi không thể tìm thấy ngữ cảnh liên quan đến câu hỏi của bạn trong đồ thị kiến thức. Vui lòng thử lại với câu hỏi khác hoặc kiểm tra lại thông tin đã được cung cấp."""
                else:
                    return f"""Sorry, I couldn't find relevant context for your question in the knowledge graph. Please try again with a different question or check the information provided."""

            # build prompt with graph context and user input
            print(f"DEBUG: Context length: {len(global_context)} characters")
            prompt = build_global_reduce_prompt(user_input, context=global_context)
            response = self.response_llm.invoke(prompt)

            return response.content, []
        elif question_type == "local":
            raw_results = self.local_search(user_input)
            graph_context = build_local_context_from_result(raw_results)

            if not graph_context:
                if is_vietnamese(user_input):
                    return f"""Xin lỗi, tôi không thể tìm thấy ngữ cảnh liên quan đến câu hỏi của bạn trong đồ thị kiến thức. Vui lòng thử lại với câu hỏi khác hoặc kiểm tra lại thông tin đã được cung cấp."""
                else:
                    return f"""Sorry, I couldn't find relevant context for your question in the knowledge graph. Please try again with a different question or check the information provided."""
            
            # build prompt with graph context and user input
            print(f"DEBUG: Context length: {len(graph_context)} characters")
            prompt = build_local_response_prompt(user_input, context=graph_context)
            response = self.response_llm.invoke(prompt)

            return response.content, raw_results
        else:
            response = self.response_llm.invoke(user_input)
            return response.content, []
        
    def process_single_chunk(self, chunk):
        # Prepare Document and Chunk Data
        doc_source = chunk.metadata.get("source", "unknown")
        doc_name = chunk.metadata.get("filename") or os.path.basename(doc_source)

        chunk_text = chunk.page_content
        chunk_page = chunk.metadata.get("page", 0) + 1
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
                    SET c.text = $chunk_text, c.embedding = $chunk_embedding, c.page = $chunk_page
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
                        "chunk_page": chunk_page,
                        "chunk_embedding": chunk_embedding,
                        "entities": batch_entities,
                        "rels": relationships
                    })

                # If write_transaction fails, the session.execute_write will rollback automatically
                session.execute_write(write_transaction)
        except Exception as e:
            print(f"Lỗi chunk (Đã rollback): {e}", flush=True)

    def process_community_worker(self, record, level):
        community_group_id = record['communityId']
        cluster = record['cluster_entities']
        
        # Bỏ qua các cụm quá nhỏ
        if len(cluster) < 4:
            return None
        
        community_id = f"COMM_L{level}_{community_group_id}"

        # Chuẩn bị dữ liệu cho LLM
        context_str = "\n".join([f"- {c['entity_name']}: {c['entity_desc']}" for c in cluster])

        try:
            chain = self.leaf_chain if level == 0 else self.parent_chain
            response = chain.invoke({"community_data": context_str})

            # clean up response
            content = response.content.replace("```json", "").replace("```", "").strip()
            info = json.loads(content)

            return {
                "community_id": community_id,
                "title": info.get("title", f"Cụm {community_group_id}"),
                "summary": info.get("summary", "unknown"),
                "full_content": info.get("full_content", "unknown"),
                "weight": len(cluster),
                "entity_names": [c['entity_name'] for c in cluster],
                "level": level
            }
        except Exception as e:
            print(f"Lỗi LLM tại community {community_id}: {e}", flush=True)
            return None
    
    def summarize_parent_batch(self, batch, level, batch_idx):
            """Hàm hỗ trợ gọi LLM tóm tắt cho một nhóm cụm con"""
            parent_id = f"COMM_L{level}_{batch_idx}"
            context_str = "\n".join([f"- Cụm {c['title']}: {c['summary']}" for c in batch])

            try:
                response = self.parent_chain.invoke({"community_data": context_str})
                
                # clean up
                content = response.content.replace("```json", "").replace("```", "").strip()
                parent_info = json.loads(content)
                
                return {
                    "parent_id": parent_id,
                    "title": parent_info.get("title", f"Tổng hợp Level {level}"),
                    "level": level,
                    "summary": parent_info.get("summary", "unknown"),
                    "full_content": parent_info.get("full_content", "unknown"),
                    "weight": len(batch),
                    "child_ids": [c['id'] for c in batch]
                }
            except Exception as e:
                print(f"Lỗi khi tóm tắt Parent Community {parent_id}: {e}", flush=True)
                return None