# Advanced RAG PDF Chatbot

A high-performance Retrieval-Augmented Generation (RAG) chatbot that lets you upload PDF documents and ask natural language questions about their content. Built with GPU-accelerated embeddings, FAISS vector search, and Ollama LLM inference.

## Project Preview

```
┌─────────────────────────────────────────────────────────────┐
│                      RAG Chatbot UI                        │
├────────────────────────┬────────────────────────────────────┤
│  ┌──────────────────┐  │  ┌──────────────────────────────┐  │
│  │  Document Upload  │  │  │  Performance Metrics         │  │
│  │  [Drag & Drop]    │  │  │  ┌─────┐                     │  │
│  │  [Process Btn]    │  │  │  │Gauge│                     │  │
│  └──────────────────┘  │  │  └─────┘                     │  │
│                        │  │  ┌──────────────────────────┐ │  │
│  ┌──────────────────┐  │  │  │  Processing Breakdown    │ │  │
│  │  Chat Interface   │  │  │  │  ████████░░░░░░░░░░░░   │ │  │
│  │  ┌──────────────┐│  │  │  └──────────────────────────┘ │  │
│  │  │ Q: What is...││  │  └──────────────────────────────┘  │
│  │  │ A: The doc.. ││  │                                    │
│  │  └──────────────┘│  │  ┌──────────────────────────────┐  │
│  │  [Input] [Send]  │  │  │  System Dashboard            │  │
│  └──────────────────┘  │  │  GPU Status │ Ollama Status   │  │
└────────────────────────┴──┴──────────────────────────────┘──┘
```

## Features

- **PDF Upload & Processing** — Drag-and-drop multiple PDFs; concurrent multi-threaded parsing with PyMuPDF
- **Intelligent Chunking** — Recursive text splitting with configurable chunk size and overlap
- **GPU-Accelerated Embeddings** — Uses sentence-transformers with CUDA support for fast vector generation
- **FAISS Vector Store** — High-performance similarity search with score threshold filtering
- **Ollama LLM Integration** — Local LLM inference using Ollama (supports Mistral, LLaMA 2, and other models)
- **Source Attribution** — Every answer cites the source document and page for verification
- **Confidence Scoring** — Average similarity score displayed per response
- **Real-Time Progress** — Live progress bar and status updates during document processing
- **Performance Dashboard** — GPU memory monitoring, chunks/sec gauge, and processing breakdown charts
- **Chat History** — Persistent conversation log with export to JSON
- **Embedding Caching** — MD5-based cache avoids recomputing identical embeddings
- **Cross-Platform** — Works on Windows, Linux, and macOS with or without GPU

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User (Streamlit Browser)                    │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                        app.py (Frontend)                            │
│  - Streamlit UI             - Chat Interface                        │
│  - File Upload Handler      - Session Management                    │
│  - Metrics Dashboard        - Export / Download                     │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                    rag_backend.py (Backend)                         │
│                                                                    │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────┐   │
│  │ FastPDFProcessor│──│ VectorStore    │──│  RAGChain          │   │
│  │ - Multi-thread  │  │ Manager        │  │  - RetrievalQA     │   │
│  │   PDF loading   │  │ - Chunking     │  │  - Source docs     │   │
│  │ - Metadata      │  │ - FAISS index  │  │  - Prompt template │   │
│  └────────────────┘  └───────┬────────┘  └─────────┬──────────┘   │
│                              │                     │               │
│  ┌────────────────┐  ┌───────▼────────┐  ┌────────▼──────────┐   │
│  │ GPUManager     │  │ EmbeddingMgr   │  │ OllamaManager     │   │
│  │ - CUDA check   │  │ - HF models    │  │ - LLM inference   │   │
│  │ - Memory info  │  │ - Cache layer  │  │ - Status check    │   │
│  └────────────────┘  └────────────────┘  └───────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────────┐
│   FAISS      │   │ Embedding    │   │    Ollama         │
│   Index      │   │ Cache        │   │    Server         │
│ (faiss_index)│   │ (pickle)     │   │ (localhost:11434) │
└──────────────┘   └──────────────┘   └──────────────────┘
```

### Data Flow

1. **Upload** → User uploads PDF files through the Streamlit UI
2. **Parse** → `FastPDFProcessor` loads PDFs concurrently using PyMuPDF
3. **Chunk** → `RecursiveCharacterTextSplitter` splits documents into overlapping chunks
4. **Embed** → `HuggingFaceEmbeddings` (sentence-transformers) generates vector embeddings with GPU acceleration
5. **Index** → FAISS indexes the vectors and saves to disk
6. **Query** → User asks a question → embedded → FAISS similarity search → retrieved chunks
7. **Generate** → Chunks + question sent to Ollama LLM → answer with source citations

## Software Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Frontend | Streamlit | Web UI framework |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Text to vector conversion |
| Vector Store | FAISS (CPU/GPU) | Similarity search |
| LLM | Ollama (Mistral / LLaMA 2) | Answer generation |
| Document Parsing | PyMuPDF | PDF text extraction |
| Text Splitting | LangChain RecursiveCharacterTextSplitter | Document chunking |
| Backend Framework | LangChain / LangChain-Classic | RAG chain orchestration |
| GPU Support | PyTorch + CUDA | Hardware acceleration |
| Visualization | Plotly | Performance charts |
| Caching | Pickle / MD5 | Embedding deduplication |
| Async I/O | ThreadPoolExecutor | Concurrent PDF processing |

## Requirements

- **Python 3.10+** (tested on 3.14)
- **Ollama** — Local LLM server ([install guide](https://ollama.ai/download))
- **4GB+ RAM** (8GB+ recommended)
- **GPU** optional but recommended for faster embeddings

### Python Dependencies

```
streamlit>=1.28.0
langchain>=0.1.0
langchain-community>=0.0.20
langchain-text-splitters>=0.0.1
torch>=2.0.0
sentence-transformers>=2.2.0
faiss-cpu>=1.7.4
PyMuPDF>=1.23.0
plotly>=5.15.0
pandas>=2.0.0
requests>=2.31.0
tiktoken>=0.5.0
```

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Sagar-R26/AI-PDF-ChatBot-RAG.git
cd AI-PDF-ChatBot-RAG
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

For GPU support, replace `faiss-cpu` with `faiss-gpu`:

```bash
pip uninstall faiss-cpu -y
pip install faiss-gpu
```

### 3. Install and Start Ollama

**Windows:** Download from [ollama.ai/download](https://ollama.ai/download)

**Linux / macOS:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Pull a model:**
```bash
# Recommended (small, fast)
ollama pull mistral

# Alternative (larger, more capable)
ollama pull llama2
ollama pull llama3.1
```

**Start Ollama:**
```bash
ollama serve
```

> **Note:** Ollama defaults to `http://localhost:11434`. The app connects here automatically.

### 4. Run the Application

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

### 5. Usage

1. Upload one or more PDF files using the drag-and-drop area
2. (Optional) Adjust chunk size, overlap, and worker count in "Processing Config"
3. Click **Process Documents** — progress is shown in real time
4. Once processing completes, type questions in the chat input
5. View answers with confidence scores and source document citations
6. Export chat history or processing stats from the sidebar

### 6. Configuration

Key settings in the sidebar "Advanced Settings" expander:

| Setting | Default | Description |
|---------|---------|-------------|
| Chunk Size | 500 | Characters per chunk (larger = more context, smaller = more precision) |
| Chunk Overlap | 50 | Overlap between chunks to maintain context boundaries |
| Max Workers | CPU count | Threads for parallel PDF loading |
| Batch Size | 100 | Documents per embedding batch |

## Author

**Sagar R**
- GitHub: https://github.com/Sagar-R26
- Project: AI PDF ChatBot (RAG)
