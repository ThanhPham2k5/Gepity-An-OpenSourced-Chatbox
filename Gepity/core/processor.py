import os
import tempfile
from langchain_community.document_loaders import PDFPlumberLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_docs_from_uploaded_files(uploaded_files: list):
    all_docs = []
    
    for uploaded_file in uploaded_files:
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        
        try:
            if suffix.lower() == ".pdf":
                loader = PDFPlumberLoader(tmp_file_path)
            elif suffix.lower() in [".docx", ".doc"]:
                loader = Docx2txtLoader(tmp_file_path)
            else:
                continue  # Skip unsupported file types

            # Load the document and split it into chunks
            all_docs.extend(loader.load())
        finally:
            # Clean up the temporary file
            if(os.path.exists(tmp_file_path)):
                os.unlink(tmp_file_path)

    return all_docs

def split_docs_into_chunks(docs: list, chunk_size=1000, chunk_overlap=100):
    # define a text splitter to split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    all_chunks = text_splitter.split_documents(docs)
    return all_chunks