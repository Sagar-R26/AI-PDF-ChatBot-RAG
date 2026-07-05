import torch
import streamlit as st
import tempfile
import shutil
import time
import os
from pathlib import Path
import json
import plotly.graph_objects as go
import pandas as pd

from rag_backend import (
    GPUManager, ProcessingConfig, RAGChain,
    create_rag_system, load_existing_rag_system,
    OllamaManager
)

st.set_page_config(
    page_title="Advanced RAG Chatbot",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        text-align: center;
    }
    .main-header p {
        color: #f0f0f0;
        text-align: center;
        margin: 0.5rem 0 0 0;
    }
    .chat-container {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .source-doc {
        background: #f1f3f4;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.25rem 0;
    }
    .success-message {
        background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .error-message {
        background: linear-gradient(90deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'processing_stats' not in st.session_state:
    st.session_state.processing_stats = {}

st.markdown("""
<div class="main-header">
    <h1>Advanced RAG Chatbot</h1>
    <p>Lightning-fast PDF processing with GPU acceleration</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("System Dashboard")

    gpu_available, gpu_name = GPUManager.is_available()
    gpu_memory = GPUManager.get_memory_info()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("GPU Status", "Available" if gpu_available else "Not Available")
    with col2:
        st.metric("Device", gpu_name if gpu_available else "CPU")

    if gpu_available:
        memory_used = gpu_memory['allocated'] / gpu_memory['total'] * 100
        st.progress(memory_used / 100)
        st.caption(f"GPU Memory: {memory_used:.1f}% used")

    st.subheader("Ollama Status")
    ollama_manager = OllamaManager()
    ollama_status, ollama_msg = ollama_manager.check_status()

    if ollama_status:
        st.success(ollama_msg)
    else:
        st.error(ollama_msg)
        with st.expander("Ollama Setup Guide"):
            st.code("""
# Install Ollama (Linux/Mac)
curl -fsSL https://ollama.ai/install.sh | sh

# Install Ollama (Windows)
# Download from https://ollama.ai/download

# Pull model
ollama pull mistral

# Start Ollama
ollama serve
            """)

    st.subheader("Processing Config")
    with st.expander("Advanced Settings", expanded=False):
        chunk_size = st.slider("Chunk Size", 200, 2000, 500, step=50)
        chunk_overlap = st.slider("Chunk Overlap", 0, 200, 50, step=10)
        max_workers = st.slider("Max Workers", 1, 16, min(8, os.cpu_count()), step=1)
        batch_size = st.slider("Batch Size", 50, 500, 100, step=50)

    config = ProcessingConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        max_workers=max_workers,
        batch_size=batch_size
    )

    if st.session_state.processing_stats:
        st.subheader("Processing Stats")
        stats = st.session_state.processing_stats
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", stats.get('num_docs', 0))
            st.metric("Processing Time", f"{stats.get('processing_time', 0):.2f}s")
        with col2:
            st.metric("Chunks", stats.get('num_chunks', 0))
            st.metric("Chunks/sec", f"{stats.get('chunks_per_sec', 0):.1f}")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Document Upload")
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Drag and drop PDF files here or click to browse"
    )

    if uploaded_files:
        st.write("**Uploaded Files:**")
        files_df = pd.DataFrame([
            {"Filename": file.name, "Size": f"{file.size / 1024:.1f} KB", "Type": file.type}
            for file in uploaded_files
        ])
        st.dataframe(files_df, use_container_width=True)

    if uploaded_files and st.button("Process Documents", type="primary"):
        temp_paths = []
        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.read())
                temp_paths.append(tmp.name)

        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(progress: float, message: str = ""):
                progress_bar.progress(progress)
                status_text.info(message)

        try:
            if os.path.exists("faiss_index"):
                shutil.rmtree("faiss_index", ignore_errors=True)

            start_time = time.time()

            st.session_state.rag_system = create_rag_system(
                temp_paths,
                config,
                progress_callback=update_progress
            )

            processing_time = time.time() - start_time
            metadata = st.session_state.rag_system.vector_manager.get_metadata()

            st.session_state.processing_stats = {
                'num_docs': metadata.get('num_docs', 0),
                'num_chunks': metadata.get('num_chunks', 0),
                'processing_time': processing_time,
                'chunks_per_sec': metadata.get('num_chunks', 0) / processing_time if processing_time > 0 else 0,
                'files': metadata.get('files', [])
            }

            progress_bar.empty()
            status_text.empty()

            st.markdown(f"""
            <div class="success-message">
                <h3>Processing Complete!</h3>
                <p>Processed {metadata.get('num_docs', 0)} documents into {metadata.get('num_chunks', 0)} chunks in {processing_time:.2f} seconds</p>
                <p>Rate: {metadata.get('num_chunks', 0) / processing_time:.1f} chunks per second</p>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.markdown(f"""
            <div class="error-message">
                <h3>Processing Failed</h3>
                <p>{str(e)}</p>
            </div>
            """, unsafe_allow_html=True)
        finally:
            for temp_path in temp_paths:
                try:
                    os.unlink(temp_path)
                except:
                    pass

with col2:
    st.subheader("Performance Metrics")
    if st.session_state.processing_stats:
        stats = st.session_state.processing_stats
        fig = go.Figure()
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=stats.get('chunks_per_sec', 0),
            title={'text': "Chunks/sec"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "#667eea"},
                'steps': [
                    {'range': [0, 25], 'color': "#ffcccc"},
                    {'range': [25, 50], 'color': "#ffffcc"},
                    {'range': [50, 100], 'color': "#ccffcc"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 75
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

        processing_data = {
            'Stage': ['PDF Loading', 'Text Splitting', 'Embedding', 'Indexing'],
            'Time (s)': [
                stats.get('processing_time', 0) * 0.3,
                stats.get('processing_time', 0) * 0.2,
                stats.get('processing_time', 0) * 0.4,
                stats.get('processing_time', 0) * 0.1
            ]
        }
        fig2 = go.Figure(data=[
            go.Bar(
                x=processing_data['Stage'],
                y=processing_data['Time (s)'],
                marker_color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
            )
        ])
        fig2.update_layout(title="Processing Breakdown", xaxis_title="Stage", yaxis_title="Time (seconds)", height=300)
        st.plotly_chart(fig2, use_container_width=True)

if st.session_state.rag_system:
    st.subheader("Chat with Your Documents")
    chat_container = st.container()

    with chat_container:
        for i, (question, answer, sources) in enumerate(st.session_state.chat_history):
            with st.expander(f"Q{i+1}: {question[:50]}...", expanded=False):
                st.write("**Question:**", question)
                st.write("**Answer:**", answer)
                if sources:
                    st.write("**Sources:**")
                    for j, source in enumerate(sources[:3]):
                        st.markdown(f"""
                        <div class="source-doc">
                            <strong>Source {j+1}:</strong> {source.metadata.get('source', 'Unknown')}<br>
                            <em>{source.page_content[:100]}...</em>
                        </div>
                        """, unsafe_allow_html=True)

        with st.form("chat_form"):
            col1, col2 = st.columns([4, 1])
            with col1:
                question = st.text_input(
                    "Ask a question about your documents:",
                    placeholder="What is the main topic discussed in the documents?"
                )
            with col2:
                submit_button = st.form_submit_button("Send", type="primary")

        if submit_button and question:
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.rag_system.query(question)
                    st.write("### Answer")
                    st.write(result["result"])
                    confidence = result.get("confidence", 0.0)
                    st.progress(confidence)
                    st.caption(f"Confidence: {confidence:.2f}")
                    if result["source_documents"]:
                        st.write("### Sources")
                        for i, doc in enumerate(result["source_documents"][:3]):
                            with st.expander(f"Source {i+1}: {doc.metadata.get('source', 'Unknown')}", expanded=False):
                                st.write(f"**Page:** {doc.metadata.get('page', 'N/A')}")
                                st.write(f"**Content:** {doc.page_content[:500]}...")
                                if 'score' in doc.metadata:
                                    st.write(f"**Relevance Score:** {doc.metadata['score']:.3f}")
                    st.session_state.chat_history.append((
                        question,
                        result["result"],
                        result["source_documents"]
                    ))
                    st.rerun()
                except Exception as e:
                    st.error(f"Error processing question: {str(e)}")

        if st.session_state.chat_history and st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("System Actions")
    if st.button("Reload RAG System"):
        try:
            if os.path.exists("faiss_index"):
                config = ProcessingConfig()
                st.session_state.rag_system = load_existing_rag_system(config)
                st.success("RAG system reloaded successfully!")
            else:
                st.error("No existing index found. Please process documents first.")
        except Exception as e:
            st.error(f"Error reloading system: {str(e)}")

    if st.button("Clear All Data"):
        st.session_state.rag_system = None
        st.session_state.chat_history = []
        st.session_state.processing_stats = {}
        if os.path.exists("faiss_index"):
            shutil.rmtree("faiss_index", ignore_errors=True)
        if os.path.exists("embedding_cache"):
            shutil.rmtree("embedding_cache", ignore_errors=True)
        st.success("All data cleared!")
        st.rerun()

with col2:
    st.subheader("Export Options")
    if st.session_state.chat_history:
        chat_export = {
            "chat_history": [
                {
                    "question": q,
                    "answer": a,
                    "sources": [{"content": s.page_content, "metadata": s.metadata} for s in sources]
                }
                for q, a, sources in st.session_state.chat_history
            ],
            "export_time": time.time()
        }
        st.download_button(
            label="Download Chat History",
            data=json.dumps(chat_export, indent=2),
            file_name=f"chat_history_{int(time.time())}.json",
            mime="application/json"
        )
    if st.session_state.processing_stats:
        stats_export = {
            "processing_stats": st.session_state.processing_stats,
            "system_info": {"gpu_available": gpu_available, "gpu_name": gpu_name, "gpu_memory": gpu_memory},
            "export_time": time.time()
        }
        st.download_button(
            label="Download Stats",
            data=json.dumps(stats_export, indent=2),
            file_name=f"processing_stats_{int(time.time())}.json",
            mime="application/json"
        )

with col3:
    st.subheader("System Info")
    st.write("**Current Config:**")
    st.json({
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "max_workers": config.max_workers,
        "batch_size": config.batch_size,
        "embedding_model": config.embedding_model
    })
    st.write("**System:**")
    st.write(f"- Python: {'.'.join(map(str, __import__('sys').version_info[:3]))}")
    st.write(f"- Streamlit: {st.__version__}")
    st.write(f"- PyTorch: {torch.__version__}")
    if gpu_available:
        st.write(f"- GPU Memory: {gpu_memory.get('total', 0) / 1024**3:.1f} GB")
    else:
        st.write("- GPU: Not Available")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>Advanced RAG Chatbot | Built with Streamlit, LangChain &amp; GPU Acceleration</p>
</div>
""", unsafe_allow_html=True)
