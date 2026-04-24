from langchain_core.prompts.chat import ChatPromptTemplate
from utils import is_vietnamese


def build_router_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", """Bạn là bộ não điều hướng của hệ thống GraphRAG. Nhiệm vụ của bạn là phân biệt cực kỳ chính xác giữa truy vấn "chi tiết" (local) và "tổng quát" (global).

        Các quy tắc phân loại:

        1. "local": 
        - Hỏi về thông tin chi tiết, định nghĩa, danh sách các thành phần, hoặc cách thức hoạt động của một tính năng cụ thể.
        - Ví dụ: "Các vai trò người dùng là gì?", "Cách đăng ký tài khoản?", "Quyền của giảng viên là gì?", "Dự án A có chức năng gì?". 
        - Ghi nhớ: Ngay cả khi câu hỏi hỏi về "danh sách" (list) các thực thể trong một hệ thống, đó vẫn là LOCAL.

        2. "global":
        - Chỉ dành cho các câu hỏi yêu cầu tóm tắt toàn bộ văn bản, tìm chủ đề chính (themes), hoặc phân tích xu hướng xuyên suốt tài liệu.
        - Ví dụ: "Tài liệu này nói về cái gì?", "Tóm tắt các điểm chính của toàn bộ dự án", "Có những rủi ro nào được nhắc đến rải rác trong file này?", "Xu hướng phát triển của hệ thống là gì?".

        3. "general": Các câu chào hỏi hoặc không liên quan tài liệu.

        CHỈ trả về JSON: {{"type": "local" | "global" | "general", "reason": "lý do"}}.
        """),
        ("human", "Câu hỏi: {user_input}")
    ])

def build_leaf_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", """Bạn là chuyên gia phân tích dữ liệu đồ thị. Nhiệm vụ của bạn là tóm tắt một cụm thông tin dựa trên danh sách thực thể được cung cấp.

        Mục tiêu: Tạo ra một văn bản phân tích mà Global Search có thể sử dụng để hiểu sâu về cụm này.

        Yêu cầu về nội dung:
        1. 'title': Tên ngắn gọn, khái quát được bản chất của nhóm.
        2. 'summary': 3-4 câu tóm tắt cốt lõi.
        3. 'full_content': Đây là phần quan trọng nhất. Hãy viết một bài phân tích chi tiết. 
        - BẮT BUỘC lồng ghép tên các thực thể quan trọng vào bài viết.
        - Giải thích mối quan hệ giữa chúng.
        - Cuối đoạn, BẮT BUỘC PHẢI thêm mục "Các thực thể chính: [danh sách thực thể]" để đảm bảo Context luôn chứa từ khóa.

        Định dạng trả về: JSON Object Tiếng Việt.
        {{"title": "...", "summary": "...", "full_content": "..."}}
        Lưu ý: Không dùng markdown code blocks, không giải thích thêm."""),
        ("human", "Danh sách thực thể và mô tả:\n{community_data}")
    ])

def build_parent_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
            ("system", """Bạn là chuyên gia tổng hợp thông tin vĩ mô. Dựa trên thông tin của các cụm chủ đề con (Sub-communities), hãy tổng hợp chúng thành một cụm chủ đề cha (Parent Community).
            - 'title': Tên chủ đề vĩ mô bao trùm tất cả các cụm con.
            - 'summary': 3-4 câu tóm tắt điểm giao thoa lớn nhất giữa các cụm con.
            - 'full_content': Đoạn văn phân tích bức tranh toàn cảnh, cách các cụm con này ghép lại để tạo thành một chủ đề lớn hơn.
            - Output: BẮT BUỘC là JSON Object bằng Tiếng Việt: {{"title": "", "summary": "", "full_content": ""}}
            Tuyệt đối không dùng markdown block.
            Mọi trường(field) đều BẮT BUỘC phải có thông tin."""),
            ("human", "Thông tin các cụm con:\n{community_data}")
        ])

def build_local_response_prompt(user_input, context):
    if is_vietnamese(user_input):
        return f"""Bạn là một trợ lý AI thông minh. Nhiệm vụ của bạn là trả lời câu hỏi dựa trên ngữ cảnh được trích xuất từ Đồ thị Tri thức (Knowledge Graph).
        
        Ngữ cảnh được cung cấp bao gồm 3 phần từ khái quát đến chi tiết:
        1. CÁC CHỦ ĐỀ LIÊN QUAN: Tóm tắt bức tranh toàn cảnh về các chủ đề/cộng đồng liên quan.
        2. THỰC THỂ CHÍNH: Các đối tượng, khái niệm quan trọng có trong câu hỏi.
        3. CHI TIẾT VĂN BẢN: Các trích đoạn nguyên bản từ tài liệu gốc, kèm theo tên file tài liệu mà đoạn văn thuộc về.

        QUY TẮC BẮT BUỘC:
        - ƯU TIÊN TỐI ĐA việc sử dụng thông tin trong ngữ cảnh được cung cấp.
        - Hãy kết hợp thông tin bao quát từ "Chủ đề liên quan" và số liệu cụ thể từ "Chi tiết văn bản" để tạo ra một câu trả lời toàn diện, logic.
        - Nếu bạn lấy thông tin trực tiếp từ phần "Chi tiết văn bản", bạn BẮT BUỘC phải trích dẫn Nguồn, kèm tên file tài liệu (Ví dụ: Theo [Nguồn 1] của tài liệu...).
        - Câu hỏi có thể có tên của tài liệu trong đó (Ví dụ (Giả sử tên file của tài liệu là "A"): "Theo thông tin của tài liệu A...", "Dựa vào file A...", hay "Dùng thông tin cung cấp từ tài liệu A..."), hãy sử dụng thông tin này kèm theo nguồn gốc tài liệu được cung cấp trong ngữ cảnh để trả lời câu hỏi.
        - NẾU NGỮ CẢNH KHÔNG CÓ CÂU TRẢ LỜI: Hãy nói "Câu hỏi cung cấp quá ít thông tin để có câu trả lời tốt.", rồi sau đó dùng kiến thức có sẵn của bạn để trả lời.

        Ngữ cảnh:
        {context}

        Câu hỏi: {user_input}

        Câu trả lời BẰNG TIẾNG VIỆT (Trình bày mạch lạc, dễ hiểu):"""
    else: 
        return f"""You are an intelligent AI assistant. Your task is to answer the user's question based on the context extracted from a Knowledge Graph.

        The provided context consists of 3 levels of information, from macro to micro:
        1. CÁC CHỦ ĐỀ LIÊN QUAN (Community Insights): Macro-level summaries of relevant topics and communities.
        2. THỰC THỂ CHÍNH (Key Entities): Important entities and concepts.
        3. CHI TIẾT VĂN BẢN (Text Chunks): Raw excerpts from the source documents, with the document's file name.

        STRICT RULES:
        - PRIORITIZE the provided context above all else.
        - Synthesize the broad understanding from the "Community Insights" with the specific details from the "Text Chunks" to provide a comprehensive answer.
        - If you use specific information from the "Text Chunks", you MUST cite your sources, and the document's file name the source came from (e.g., According to [Nguồn 1] from document...).
        - The Question could come with the document's file name (e.g. (Assume the document's name in these examples is "A"), "According to the information from file A...", "Based on document A...", or "Use the informations provided by document A..." ), use this information with the document's file name from the context to answer the question.
        - IF THE CONTEXT DOES NOT CONTAIN THE ANSWER: You must state "The question provided too little information to form a good answer.", then use your available knowledge to answer.

        Context:
        {context}

        Question: {user_input}

        Answer IN ENGLISH (Clear, concise, and well-structured):"""

def build_global_prompt(user_input, context):
    if is_vietnamese(user_input):
        return f"""Bạn là một trợ lý AI thông minh. Nhiệm vụ của bạn là tổng hợp các báo cáo trung gian để tạo ra câu trả lời toàn diện nhất cho câu hỏi của người dùng.
        
        Ngữ cảnh được cung cấp bên dưới là TỔNG HỢP CÁC PHÂN TÍCH TOÀN CỤC (Intermediate Answers) được trích xuất từ nhiều cộng đồng dữ liệu khác nhau, kèm theo tên file tài liệu mà báo cáo đó thuộc về. Nó mang tính chất bao quát và đa chiều.

        QUY TẮC BẮT BUỘC:
        - ƯU TIÊN TỐI ĐA việc sử dụng thông tin trong ngữ cảnh được cung cấp. Không bịa đặt thêm dữ liệu.
        - Hãy xâu chuỗi các điểm dữ liệu rời rạc từ các báo cáo khác nhau để tạo thành một bức tranh toàn cảnh, mạch lạc và có tính logic cao.
        - Câu hỏi có thể có tên của tài liệu trong đó (Ví dụ (Giả sử tên file của tài liệu là "A"): "Theo thông tin của tài liệu A...", "Dựa vào file A...", "Tài liệu A đang nói về chủ đề gì?", "Tóm tắt các nội dung của tài liệu A"), hãy sử dụng thông tin này kèm theo nguồn gốc tài liệu được cung cấp trong ngữ cảnh để trả lời câu hỏi.
        - Bỏ qua các thông tin mâu thuẫn hoặc không đóng góp trực tiếp vào việc trả lời câu hỏi.
        - NẾU NGỮ CẢNH KHÔNG CÓ CÂU TRẢ LỜI: Hãy nói "Dữ liệu toàn cục hiện tại không đủ để đưa ra câu trả lời cho câu hỏi này.".

        Ngữ cảnh (Các báo cáo trung gian):
        {context}

        Câu hỏi: {user_input}

        Câu trả lời BẰNG TIẾNG VIỆT (Trình bày mạch lạc, dễ hiểu):"""
    else: 
        return f"""You are an intelligent AI assistant. Your task is to synthesize intermediate reports to create the most comprehensive answer to the user's question.

        The provided context below contains SYNTHESIZED GLOBAL ANALYSES (Intermediate Answers) extracted from various data communities, and the documentt's file name the reports came from. It is broad and multi-dimensional.

        STRICT RULES:
        - PRIORITIZE the provided context above all else. Do not fabricate data.
        - Connect disparate data points from different reports to form a coherent, highly logical, and big-picture response.
        - The Question could come with the document's file name (e.g. (Assume the document's name in these examples is "A"), "According to the information from file A...", "Based on document A...", "Use the informations provided by document A...", "What is document A talking about?", "Summarize document A content"), use this information with the document's file name from the context to answer the question.
        - Ignore conflicting information or details that do not directly contribute to answering the question.
        - IF THE CONTEXT DOES NOT CONTAIN THE ANSWER: You must state "The current global data is insufficient to provide an answer to this question.".

        Context (Intermediate Reports):
        {context}

        Question: {user_input}

        Answer IN ENGLISH (Clear, concise, well-structured):"""

def build_local_context_from_result(results):
    if len(results) == 0:
        return ""
    all_chunks_with_source = {}
    all_entities = set()
    all_summaries = set()

    for res in results:
        chunk_source = res['source_file']
        if chunk_source not in all_chunks_with_source:
            all_chunks_with_source[chunk_source] = []
        all_chunks_with_source[chunk_source].append(res['text'])

        if res['related_entities']:
            all_entities.update(res['related_entities'])

        if res['related_community_summaries']:
            all_summaries.update(res['related_community_summaries'])

    # build the context
    final_context = "Dưới đây là các thông tin ngữ cảnh được trích xuất từ cơ sở dữ liệu tri thức:\n\n"

    # Community Insights
    if all_summaries:
        final_context += "### CÁC CHỦ ĐỀ LIÊN QUAN:\n"
        final_context += "\n".join([f"- {s}" for s in all_summaries]) + "\n\n"

    # Entities
    if all_entities:
        final_context += f"### THỰC THỂ CHÍNH: {', '.join(all_entities)}\n\n"

    # Chunks
    final_context += "### CHI TIẾT VĂN BẢN:\n"
    for source, text_list in all_chunks_with_source.items():
        final_context += f"[Các nguồn từ tài liệu: {source}]:\n\n"
        for i, text in enumerate(text_list, 1):
            final_context += f"[Nguồn {i}]: {text}\n\n"

    return final_context

def build_global_context_from_result(results):
    if len(results) == 0:
        return ""
    
    # build the context
    final_context = "Dưới đây là các thông tin ngữ cảnh được trích xuất từ cơ sở dữ liệu tri thức:\n\n"

    for i, res in enumerate(results):
        final_context += f"--- BÁO CÁO CỘNG ĐỒNG {i+1} [NGUỒN TÀI LIỆU: {res['source']}] ---\n{res['full_content']}\n\n"

    return final_context