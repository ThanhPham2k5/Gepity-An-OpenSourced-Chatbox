# 🤖 Gepity — An Open-Sourced Chatbox

> Upload a PDF or DOCX file and ask questions about its content — powered by local LLMs via Ollama.

---

## ✨ Features

- 📄 Upload **PDF / DOCX** documents
- 💬 Ask questions in **Vietnamese or English**
- 🔍 **Basic RAG** Semantic search with **FAISS** vector store & **GraphRAG** powered by **Neo4j**
- 🧠 Runs fully **local** with **Qwen2.5:7b** via Ollama — no API key needed
- ⚡ Built with **Streamlit** for a fast, clean UI

---

## 🚀 Getting Started

### Prerequisites

- [WSL (Ubuntu)](https://learn.microsoft.com/en-us/windows/wsl/install) or Linux/macOS
- Python 3.8+
- [Ollama](https://ollama.ai) installed on your machine
- [Neo4j Desktop](https://neo4j.com/download/) installed and running

---

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/ThanhPham2k5/Gepity-An-OpenSourced-Chatbox.git
cd Gepity-An-OpenSourced-Chatbox
```

**2. Create and activate a virtual environment**

```bash
python3 -m venv myenv
source myenv/bin/activate
```

**3. Install dependencies**

```bash
cd Gepity/
make install
```

**4. Pull the LLM model** _(requires Ollama to be installed)_

```bash
sudo snap install ollama
ollama pull qwen2.5:3b
ollama list
ollama serve
```

---

### Setting up Neo4j for GraphRAG
To utilize the Knowledge Graph capabilities, you need to set up a local Neo4j database:

1. Download and install [Neo4j Desktop](https://neo4j.com/download/).

2. Open Neo4j Desktop and create a **New Instance**.

3. Add a **Local DBMS** to your project (set a password you will remember).

4. Click **Start** on your new DBMS to run the database.

5. Take note of the Bolt/Neo4j port (usually 7687) and your password.

---

### Environment Variables Setup
You need to configure the .env file to connect the application to your Neo4j database and HuggingFace API.

1. In the root directory, locate the [.env_example](./.env_example) file.

2. Create a copy of it and name it .env:

```bash
cp .env_example .env
```

3. Open the .env file and update the values with your actual database credentials and token:

```
# Example .env configuration
NEO4J_URI="bolt://localhost:7687" # Update IP if running inside WSL to Windows host
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="your-neo4j-password"
NEO4J_DATABASENAME="neo4j"

HF_TOKEN="YOUR-HUGGING-FACE-TOKEN"
```
(Note: If you are running WSL and Neo4j is on Windows, use ip route | grep default to find the correct IP address instead of localhost).

---

### Running the App

In your terminal (make sure your virtual environment is activated and Neo4j is running):

```bash
make run
```

Then open your browser at `http://localhost:8501`

---

## 🗂️ Project Structure

```
Gepity-An-OpenSourced-Chatbox/
├── .env                        # Environment variables (create from .env_example)
├── .env_example                # Template for environment variables
├── .gitignore
├── LICENSE
├── README.md
├── script_connection.py        # Script to verify Neo4j database connection
├── tests/                      # Directory for pytest test cases
├── Gepity/                     # Main Application Directory
│   ├── .streamlit/             # Streamlit UI configuration
│   ├── assets/                 # Static assets (images, icons)
│   ├── core/                   # Core engine logic
│   │   ├── __init__.py
│   │   ├── engine.py           # Main BasicRAG implementation
│   │   ├── graph_rag_engine.py # Neo4j GraphRAG implementation
│   │   └── processor.py        # Document parsing and chunking
│   ├── styles/                 # UI Styling components
│   ├── utils/                  # Helper functions
│   │   ├── __init__.py
│   │   └── helpers.py
│   ├── app.py                  # Main Streamlit application entry point
│   ├── database.py             # Database connection handlers
│   ├── demo.css                # Demo styling
│   ├── style.css               # Global CSS styling
│   ├── Makefile                # Shortcuts for common commands
│   ├── requirements.txt        # Python dependencies
│   └── script_graph_logic.py   # Script for testing graph querying logic
```

---

## 🛠️ Makefile Commands

| Command           | Description                      |
| ----------------- | -------------------------------- |
| `make install`    | Install Python dependencies      |
| `make run`        | Start the Streamlit app          |
| `make pull-model` | Download Qwen2.5:7b via Ollama   |
| `make clean`      | Remove cache and generated files |

---

## 📚 Tech Stack

| Layer            | Technology                                     |
| ---------------- | ---------------------------------------------- |
| Frontend         | Streamlit                                      |
| LLM Runtime      | Ollama + Qwen2.5                               |
| Framework        | LangChain (Core, Neo4j, Experimental, Ollama)  |
| Knowledge Graph  | Neo4j, neo4j-graphrag                          |
| Embeddings       | sentence-transformers (multilingual MPNet)     |
| Vector Store     | FAISS                                          |
| Document Parsing | PDFPlumber, PyPDF, docx2txt                    |
| NLP & NER        | NLTK, GLiNER                                   |
| Keyword Search   | rank_bm25                                      |
| Testing          | Pytest                                         |


---

## 📝 License

This project is licensed under the terms of the [LICENSE](./LICENSE) file.
