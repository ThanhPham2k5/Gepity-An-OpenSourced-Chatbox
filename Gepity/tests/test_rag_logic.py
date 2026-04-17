from Gepity.app import get_docs_from_uploaded_files as GetDocsFromUpload
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from io import BytesIO
from reportlab.pdfgen import canvas

def test_chunking_logic():
    # create sample documents
    raw_texts = [
        "this is a sample document that is quite long and needs to be split into smaller chunks for better processing by the language model. " * 50, # 1000 characters
    ]

    documents = [Document(page_content=text) for text in raw_texts]

    #init text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )

    # split documents into chunks
    chunks = text_splitter.split_documents(documents)

    # assertions to verify the chunking logic
    # assert that the chunks are created correctly
    assert len(chunks) > 0

    # assert that each chunk is less than or equal to the specified chunk size
    assert all(len(chunk.page_content) <= 1000 for chunk in chunks)

    # Check if the last 20 characters of a chunk appear in the next chunk
    for i in range(len(chunks) - 1):
        tail = chunks[i].page_content[-20:]
        assert tail in chunks[i+1].page_content

    # assert that the total length of all chunks is greater or equal to the original document length (due to overlap)
    assert sum(len(chunk.page_content) for chunk in chunks) >= sum(len(doc.page_content) for doc in documents)


def test_upload_files():
    # create a valid, minimal PDF in memory
    buffer = BytesIO()
    c = canvas.Canvas(buffer)
    c.drawString(100, 750, "This is real PDF content for testing.")
    c.save()
    sample_pdf_content = buffer.getvalue()

    class MockFile:
        def __init__(self, name, content):
            self.name = name
            self.content = content
        def read(self):
            return self.content

    mock_file = MockFile("sample.pdf", sample_pdf_content)

    # call the GetDocsFromUpload function with the mock file
    documents = GetDocsFromUpload([mock_file])

    # assertions to verify the document extraction logic
    # assert that documents are extracted correctly from the uploaded file
    assert len(documents) > 0

    # assert that the extracted documents are of the correct type and contain expected content
    assert all(isinstance(doc, Document) for doc in documents)
    assert all(isinstance(doc.page_content, str) for doc in documents)
    assert "testing" in documents[0].page_content