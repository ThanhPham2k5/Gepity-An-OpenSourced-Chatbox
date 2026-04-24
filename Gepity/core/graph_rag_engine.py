from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from gliner import GLiNER
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from .processor import get_docs_from_uploaded_files, split_docs_into_chunks, extract_and_link_entities
from .builder import build_local_response_prompt, build_global_prompt, build_router_prompt, build_leaf_prompt, build_parent_prompt, build_global_context_from_result, build_local_context_from_result
from utils import is_vietnamese, setup_constraints, extract_json_from_response
from pathlib import Path

from database import get_graph_connection
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx
from dotenv import load_dotenv
load_dotenv()

def _is_running_in_streamlit():
    return get_script_run_ctx() is not None

# Use for Windows IP
WINDOWS_IP = "172.25.64.1"

# Use for WSL IP
# WINDOWS_IP = "localhost"
MODELS_CACHE = "../../models_cache"

class Graph_engine:
    def __init__(self, summary_model_name="qwen2.5:3b", response_model_name="qwen2.5:3b"):
        
        # use to response 
        self.response_llm = ChatOllama(
            model=response_model_name, 
            temperature=0.6, # higher temperature for more creative response generation
            num_ctx=8192,
            base_url=f"http://{WINDOWS_IP}:11434"
        )

        # use to determine what type of question the user is asking
        self.router_llm = ChatOllama(
            model="qwen2.5:3b",
            temperature=0,          
            format="json",            
            num_ctx=2048,             
            num_thread=4,
            base_url=f"http://{WINDOWS_IP}:11434"
        )

        # use to build community summary
        self.summary_llm = ChatOllama(
            model=summary_model_name,
            temperature=0,
            num_ctx=6144,
            format="json",
            base_url=f"http://{WINDOWS_IP}:11434"
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

        os.environ['HF_HOME'] = MODELS_CACHE
        self.embedder = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            model_kwargs={"device": "cpu", "token": os.getenv("HF_TOKEN")},
            encode_kwargs={"normalize_embeddings": True},
            cache_folder= MODELS_CACHE
        )

        self.gliner_model = GLiNER.from_pretrained("urchade/gliner_multi")
        

    def process_document(self, uploaded_files, chunk_size, chunk_overlap):
        # read files and extract text
        all_docs = get_docs_from_uploaded_files(uploaded_files)

        # Tạo một dict để chứa chunks theo từng file
        # Key: tên file (source), Value: danh sách chunks
        docs_with_chunks = {}

        for doc in all_docs:
            source_name = doc.metadata.get('source', 'unknown_doc')
            source_without_suffix = Path(source_name).stem
            chunks = split_docs_into_chunks([doc], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            if source_without_suffix not in docs_with_chunks:
                docs_with_chunks[source_without_suffix] = []
        
            docs_with_chunks[source_without_suffix].extend(chunks)

        return docs_with_chunks

    def sync_to_graph(self, all_chunks, source):
        # TODO: add processed document list to avoid reprocessing the same document which leads to bloat and noise

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
            st.info(f"Đang xử lý {total_chunks} chunks...")
            status_text = st.empty()
        else:
            print(f"Đang xử lý {total_chunks} chunks...")

        # Use ThreadPoolExecutor with as_completed for real-time tracking
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Submit all tasks
            future_to_chunk = {executor.submit(self.process_single_chunk, chunk, doc_source=source): chunk for chunk in all_chunks}
            
            for future in as_completed(future_to_chunk):
                completed_count += 1
                progress_pct = completed_count / total_chunks
                
                # Update UI
                if in_streamlit:
                    status_text.text(f"Tiến độ: {completed_count}/{total_chunks} chunks hoàn tất")
                else:
                    # Update console
                    if completed_count % 10 == 0:
                        print(f"Tiến độ: [{completed_count}/{total_chunks}] - {progress_pct:.1%}", flush=True)

                try:
                    future.result() 
                except Exception as e:
                    print(f"Error in a thread: {e}")

        if in_streamlit:
            st.success("Đã hoàn thành xử lý toàn bộ chunks!")
        else:
            print("Đã hoàn thành xử lý toàn bộ chunks!")

        # create vector index for graph
        # self.create_vector_indexes()
        self.build_community(source)

    def build_community(self, doc_source):

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
        source_filter = "AND r.source = $source"
        project_cypher = f"""
        MATCH (source:Entity)-[r:RELATES_TO]->(target:Entity)
        WHERE r.weight >= 2 {source_filter}
        WITH gds.graph.project(
            '{project_name}',
            source,
            target,
            {{
                sourceNodeLabels: labels(source),
                targetNodeLabels: labels(target),
                relationshipType: type(r),
                relationshipProperties: {{ weight: r.weight }}
            }},
            {{
                undirectedRelationshipTypes: ['*'] 
            }}
        ) AS g
        RETURN g.graphName AS graphName, g.nodeCount AS nodes, g.relationshipCount AS rels
        """
        
        try:
            self.graph.query(
                project_cypher, {"source": doc_source}
            )
        except Exception as e:
            print(f"Lỗi khi tạo Projection: {e}")
            return

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

        for record in results:
            result = self.process_community_worker(record, 0, doc_source)
            if result:
                processed_communities.append(result)
                msg = f"Tiến độ: đã xử lý {len(processed_communities)} community"
                print(msg)

        # single batch query
        if processed_communities:
            save_batch_cypher = """
            UNWIND $data AS item
            MERGE (c:Community {id: item.community_id})
            SET c.title = item.title,
                c.level = item.level,
                c.summary = item.summary,
                c.full_content = item.full_content,
                c.weight = item.weight,
                c.source = $source
            WITH c, item
            UNWIND item.entity_names AS e_name
            MATCH (e:Entity {name: e_name})
            MERGE (e)-[:IN_COMMUNITY]->(c)
            """
            self.graph.query(save_batch_cypher, {
                "data": processed_communities,
                "source": doc_source
            })
        level_0_community_ids = [c['community_id'] for c in processed_communities]

        # Gọi hàm tạo Level 1 như cũ
        print("Đang xây dựng Parent Communities (Level 1)...")
        if len(level_0_community_ids) > 1:
            self.build_parent_communities(doc_source=doc_source, child_community_ids=level_0_community_ids)

        if in_streamlit:
            st.success("Hoàn tất xây dựng Community!")
        else:
            print("Hoàn tất xây dựng Community!")


    def build_parent_communities(self, doc_source, child_community_ids, current_level=1):

        print(f"Đang tổng hợp Level {current_level} Communities...")

        # Lấy summary của các child communities từ Neo4j
        fetch_cypher = """
        MATCH (c:Community) 
        WHERE c.id IN $ids AND c.source = $source
        RETURN c.id AS id, c.title AS title, c.summary AS summary
        """
        children = self.graph.query(fetch_cypher, {
            "ids": child_community_ids,
            "source": doc_source
        })
        
        if not children:
            return

        # Nhóm các child communities thành các batch
        batch_size = 5 
        
        parent_results = []

        for i in range(0, len(children), batch_size):
            batch = children[i : i + batch_size]
            result = self.summarize_parent_batch(batch, current_level, i, doc_source)
            if result:
                parent_results.append(result)

        # single batch query
        if parent_results:
            save_parent_cypher = """
            UNWIND $data AS item
            MERGE (p:Community {id: item.parent_id})
            SET p.title = item.title,
                p.level = item.level,
                p.summary = item.summary,
                p.full_content = item.full_content,
                p.weight = item.weight,
                p.source = $source
            WITH p, item
            UNWIND item.child_ids AS c_id
            MATCH (c:Community {id: c_id})
            MERGE (c)-[:PARENT_COMMUNITY]->(p)
            """
            self.graph.query(save_parent_cypher, {
                "data": parent_results,
                "source": doc_source
            })

        # Recursive: if there's more than one parent, continue to higher level
        new_parent_ids = [p['parent_id'] for p in parent_results]
        
        if len(new_parent_ids) > 1:
            self.build_parent_communities(doc_source, new_parent_ids, current_level=current_level + 1)
        else:
            msg = f"Đã đạt đến Root Community ở Level {current_level}!"
            print(msg)

    def global_search(self, source=None, level=None):

        level_source_filter = "WHERE c.source=$source" if source else ""
        if not level: # if level was not specified, get the second highest level
            level_query = f"""
            MATCH (c:Community)
            {level_source_filter}
            ORDER BY c.level DESC
            SKIP 1
            LIMIT 1
            RETURN c.level as level
            """
            level_data = self.graph.query(level_query,{"source": source} if source else {})
            level = level_data[0]['level']

        retrieval_source_filter = "AND comm.source = $source" if source else ""
        retrieval_query = f"""
        MATCH (comm:Community)
        WHERE comm.level = $level AND comm.full_content <> "" {retrieval_source_filter}
        RETURN DISTINCT comm.full_content as full_content, comm.source as source
        """

        results = self.graph.query(retrieval_query, {
            "source": source,
            "level": level
        })

        return results

    def local_search(self, user_input, doc_source=None, threshold=0.65, top_k=5):
        # embed user input for vector search
        query_vector = self.embedder.embed_query(user_input)
        
        # use cypher to query relevant chunks, entities and community summaries for said entities
        source_filter = "AND chunk.source = $doc_source" if doc_source else ""
        retrieval_query = f"""
        CALL db.index.vector.queryNodes('chunk_embeddings', $top_k, $query_vector)
        YIELD node AS chunk, score
        WHERE score >= $threshold {source_filter}

        OPTIONAL MATCH (chunk)-[:HAS_ENTITY]->(e:Entity)
        WHERE e.source = chunk.source
        OPTIONAL MATCH (e)-[:IN_COMMUNITY]->(comm:Community)
        WHERE comm.source = chunk.source
        
        RETURN
            chunk.text as text,
            chunk.page as page_number,
            chunk.source as source_file,
            score,
            collect(DISTINCT e.name) as related_entities,
            collect(DISTINCT comm.summary) as related_community_summaries
        """

        results = self.graph.query(retrieval_query, {
            "query_vector": query_vector,
            "top_k": top_k,
            "threshold": threshold,
            "doc_source": doc_source
        })
        
        return results
    
    def local_hybrid_search(self, user_input, doc_source=None, threshold=0.65, top_k=5):
        # embed user input for vector search
        query_vector = self.embedder.embed_query(user_input)
        
        # vector search
        source_filter = "AND chunk.source = $doc_source" if doc_source else ""
        vector_query = f"""
        CALL db.index.vector.queryNodes('chunk_embeddings', $top_k, $query_vector)
        YIELD node AS chunk, score
        WHERE score >= $threshold {source_filter}
        RETURN 
            elementId(chunk) AS chunk_id, 
            chunk.text AS text, 
            chunk.page AS page_number, 
            chunk.source AS source_file, 
            score AS vector_score
        """
        # get chunks from vector result
        vector_results = self.graph.query(vector_query, {
            "query_vector": query_vector,
            "top_k": top_k,
            "threshold": threshold,
            "doc_source": doc_source
        })

        # keyword search
        keyword_query = f"""
        CALL db.index.fulltext.queryNodes('chunk_text_index', $user_input, {{limit: $top_k}})
        YIELD node AS chunk, score
        WHERE 1=1 {source_filter}
        RETURN 
            elementId(chunk) AS chunk_id, 
            chunk.text AS text, 
            chunk.page AS page_number, 
            chunk.source AS source_file, 
            score AS keyword_score
        """
        # get chunks from keyword result
        keyword_results = self.graph.query(keyword_query, {
            "user_input": user_input,
            "top_k": top_k,
            "doc_source": doc_source
        })

        # combine with Reciprocal Rank Fusion
        # Công thức: RRF_score = 1 / (k + rank)
        rrf_k = 60 # constant
        combined_results = {}

        # Xử lý rank cho Vector
        for rank, res in enumerate(vector_results, start = 1):
            chunk_id = res['chunk_id']
            if chunk_id not in combined_results:
                combined_results[chunk_id] = {'data': res, 'rrf_score': 0}
            combined_results[chunk_id]['rrf_score'] += 1.0 / (rrf_k + rank)

        # Xử lý rank cho Keyword
        for rank, res in enumerate(keyword_results, start = 1):
            chunk_id = res['chunk_id']
            if chunk_id not in combined_results:
                combined_results[chunk_id] = {'data': res, 'rrf_score': 0}
            combined_results[chunk_id]['rrf_score'] += 1.0 / (rrf_k + rank)

        # Sắp xếp lại dựa trên RRF score và lấy top_k
        sorted_combined = sorted(combined_results.values(), key=lambda x: x['rrf_score'], reverse=True)[:top_k]

        final_results = []
        if sorted_combined:
            chunk_ids = [item['data']['chunk_id'] for item in sorted_combined]
            
            # get related entities and communities
            enrichment_query = """
            MATCH (chunk) WHERE elementId(chunk) IN $chunk_ids
            OPTIONAL MATCH (chunk)-[:HAS_ENTITY]->(e:Entity)
            WHERE e.source = chunk.source
            OPTIONAL MATCH (e)-[:IN_COMMUNITY]->(comm:Community)
            WHERE comm.source = chunk.source
            RETURN 
                elementId(chunk) AS chunk_id,
                collect(DISTINCT e.name) as related_entities,
                collect(DISTINCT comm.summary) as related_community_summaries
            """
            enriched_data = self.graph.query(enrichment_query, {"chunk_ids": chunk_ids})
            
            # Map enrichment data
            enrichment_map = {row['chunk_id']: row for row in enriched_data}
            
            for item in sorted_combined:
                base_data = item['data']
                c_id = base_data['chunk_id']
                enrich_info = enrichment_map.get(c_id, {'related_entities': [], 'related_community_summaries': []})
                
                final_results.append({
                    'text': base_data['text'],
                    'page_number': base_data.get('page_number'),
                    'source_file': base_data.get('source_file'),
                    'score': item['rrf_score'],
                    'related_entities': enrich_info['related_entities'],
                    'related_community_summaries': enrich_info['related_community_summaries']
                })

        return final_results
        
    def get_response(self, user_input, doc_source=None, hybrid_search=False):
        if not self.graph:
            st.error("Không thể kết nối đến Neo4j")
            if is_vietnamese(user_input):
                return "Xin lỗi, tôi không thể truy cập vào đồ thị kiến thức vào lúc này. Vui lòng thử lại sau.", []
            else:
                return "Sorry, I cannot access the knowledge graph at the moment. Please try again later.", []
        
        # get router response
        router_response = self.router_chain.invoke(user_input)

        # clean up response
        router_content = router_response.content.replace("```json", "").replace("```", "").strip()
        router_info = json.loads(router_content)

        question_type = router_info.get("type", "general")
        reason = router_info.get("reason")
        print(f"DEBUG: Question type: {question_type}")
        print(f"DEBUG: Reason: {reason}")

        if question_type == "global":
            raw_results = self.global_search(source=doc_source)
            global_context = build_global_context_from_result(raw_results)

            if not global_context:
                if is_vietnamese(user_input):
                    return "Xin lỗi, tôi không thể tìm thấy ngữ cảnh liên quan đến câu hỏi của bạn trong đồ thị kiến thức. Vui lòng thử lại với câu hỏi khác hoặc kiểm tra lại thông tin đã được cung cấp.", []
                else:
                    return "Sorry, I couldn't find relevant context for your question in the knowledge graph. Please try again with a different question or check the information provided.", []

            # build prompt with graph context and user input
            print(f"DEBUG: Context length: {len(global_context)} characters")
            prompt = build_global_prompt(user_input, context=global_context)
            response = self.response_llm.invoke(prompt)

            return response.content, []
        elif question_type == "local":
            if not hybrid_search:
                raw_results = self.local_search(user_input, doc_source)
            else:
                raw_results = self.local_hybrid_search(user_input, doc_source)

            graph_context = build_local_context_from_result(raw_results)

            if not graph_context:
                if is_vietnamese(user_input):
                    return "Xin lỗi, tôi không thể tìm thấy ngữ cảnh liên quan đến câu hỏi của bạn trong đồ thị kiến thức. Vui lòng thử lại với câu hỏi khác hoặc kiểm tra lại thông tin đã được cung cấp.", []
                else:
                    return "Sorry, I couldn't find relevant context for your question in the knowledge graph. Please try again with a different question or check the information provided." , []
            
            # build prompt with graph context and user input
            print(f"DEBUG: Context length: {len(graph_context)} characters")
            prompt = build_local_response_prompt(user_input, context=graph_context)
            response = self.response_llm.invoke(prompt)

            return response.content, raw_results
        else:
            response = self.response_llm.invoke(user_input)
            return response.content, []
        
    def process_single_chunk(self, chunk, doc_source):
        # Prepare Document and Chunk Data
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
                    MERGE (d:Document {source: $doc_source}) SET d.name = $doc_source
                    MERGE (c:Chunk {id: $chunk_id})
                    SET c.text = $chunk_text, c.embedding = $chunk_embedding, c.page = $chunk_page, c.source = $doc_source
                    MERGE (c)-[:PART_OF]->(d)

                    // Batch Entities
                    WITH c
                    UNWIND $entities AS ent
                    MERGE (e:Entity {name: ent.name})
                    SET e.description = ent.desc, e.embedding = ent.embedding, e.label = ent.label, e.source = $doc_source
                    MERGE (c)-[:HAS_ENTITY]->(e)

                    // Batch Relationships (Co Occurence)
                    WITH c
                    UNWIND $rels AS rel
                    MATCH (src:Entity {name: rel.source})
                    MATCH (tgt:Entity {name: rel.target})
                    MERGE (src)-[r:RELATES_TO]->(tgt)
                    ON CREATE SET r.weight = 1, r.type = rel.type, r.source = $doc_source
                    ON MATCH SET r.weight = r.weight + 1
                    """

                    tx.run(sync_query, {
                        "doc_source": doc_source,
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

    def process_community_worker(self, record, level, doc_source):
        community_group_id = record['communityId']
        cluster = record['cluster_entities']
        
        # Bỏ qua các cụm quá nhỏ
        if len(cluster) < 4:
            return None
        
        # clean doc_source
        clean_source = str(doc_source).replace(" ", "_").replace(".pdf", "")
        # create id with source
        community_id = f"{clean_source}_COMM_L{level}_{community_group_id}"

        # Chuẩn bị dữ liệu cho LLM
        context_str = "\n".join([f"- {c['entity_name']}: {c['entity_desc']}" for c in cluster])

        try:
            chain = self.leaf_chain if level == 0 else self.parent_chain
            response = chain.invoke({"community_data": context_str})

            info = extract_json_from_response(response.content)
            
            full_content = info.get("full_content", "").strip()
            summary = info.get("summary", "").strip()
            if not full_content or not summary:
                raise ValueError("LLM trả về rỗng")

            return {
                "community_id": community_id,
                "title": info.get("title", f"Cụm {community_group_id}"),
                "summary": summary,
                "full_content": full_content,
                "weight": len(cluster),
                "entity_names": [c['entity_name'] for c in cluster],
                "level": level
            }
        except Exception as e:
            print(f"Lỗi LLM tại {community_id}, đang sử dụng nội dung Fallback...")
            entity_names = [c['entity_name'] for c in cluster]
            top_entities = ", ".join(entity_names[:5]) # Lấy 5 tên đầu tiên
            
            return {
                "community_id": community_id,
                "title": f"Cụm thực thể hỗn hợp {community_group_id}",
                "summary": f"Cụm này bao gồm các thực thể liên quan như {top_entities}.",
                "full_content": f"Cộng đồng này được hệ thống gom nhóm tự động dựa trên mức độ xuất hiện cùng nhau trong tài liệu. Các thực thể chính đóng vai trò trung tâm bao gồm: {top_entities}. Thông tin chi tiết chưa thể tổng hợp tự động do dữ liệu phân tán.",
                "weight": len(cluster),
                "entity_names": entity_names,
                "level": level
            }
    
    def summarize_parent_batch(self, batch, level, batch_idx, doc_source):
        
        # clean doc_source
        clean_source = str(doc_source).replace(" ", "_").replace(".pdf", "")
        # create id with source
        parent_id = f"{clean_source}_COMM_L{level}_{batch_idx}"
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
            
    def empty_database(self):
        in_streamlit = _is_running_in_streamlit()

        if not self.graph:
            if in_streamlit:
                st.error("Không thể kết nối đến Neo4j sau nhiều lần thử. Vui lòng kiểm tra console.")
            else:
                print("Không thể kết nối đến Neo4j sau nhiều lần thử. Vui lòng kiểm tra console.")
            return 0

        delete_cypher = """
        MATCH (n)
        DETACH DELETE n;
        """

        self.graph.query(delete_cypher)

    def create_vector_indexes(self):
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

    def create_fulltext_index(self):
        cypher_query = """
        CREATE FULLTEXT INDEX chunk_text_index IF NOT EXISTS
        FOR (c:Chunk) ON EACH [c.text]
        """

        try:
            self.graph.query(cypher_query)
        except Exception as e:
            print(f"Index creation note: {e}")