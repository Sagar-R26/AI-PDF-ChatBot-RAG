import os
import torch
import logging
import time
import json
import pickle
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from dataclasses import dataclass

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    chunk_size: int = 500
    chunk_overlap: int = 50
    max_workers: int = mp.cpu_count()
    batch_size: int = 100
    index_dir: str = "faiss_index"
    embedding_model: str = "intfloat/e5-small"
    cache_embeddings: bool = True


class GPUManager:
    @staticmethod
    def is_available() -> tuple:
        try:
            if torch.cuda.is_available():
                return True, torch.cuda.get_device_name(0)
            return False, "No GPU found"
        except Exception as e:
            logger.error(f"GPU check failed: {e}")
            return False, str(e)

    @staticmethod
    def get_device() -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def get_memory_info() -> Dict[str, Any]:
        if torch.cuda.is_available():
            return {
                "total": torch.cuda.get_device_properties(0).total_memory,
                "allocated": torch.cuda.memory_allocated(0),
                "cached": torch.cuda.memory_reserved(0)
            }
        return {"total": 0, "allocated": 0, "cached": 0}


class EmbeddingManager:
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.device = GPUManager.get_device()
        self._model = None
        self._cache_dir = Path("embedding_cache")
        self._cache_dir.mkdir(exist_ok=True)
        logger.info(f"Using device for embeddings: {self.device}")

    @property
    def model(self):
        if self._model is None:
            self._model = HuggingFaceEmbeddings(
                model_name=self.config.embedding_model,
                model_kwargs={"device": self.device},
                encode_kwargs={"normalize_embeddings": True}
            )
        return self._model

    def _get_cache_key(self, texts: List[str]) -> str:
        content = "".join(texts)
        return hashlib.md5(content.encode()).hexdigest()

    def _load_cached_embeddings(self, cache_key: str) -> Optional[List[List[float]]]:
        if not self.config.cache_embeddings:
            return None
        cache_file = self._cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cached embeddings: {e}")
        return None

    def _save_embeddings_cache(self, cache_key: str, embeddings: List[List[float]]):
        if not self.config.cache_embeddings:
            return
        cache_file = self._cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(embeddings, f)
        except Exception as e:
            logger.warning(f"Failed to save embeddings cache: {e}")


class FastPDFProcessor:
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def process_single_pdf(self, file_path: str) -> List[Document]:
        try:
            loader = PyMuPDFLoader(file_path)
            docs = loader.load()
            filename = os.path.basename(file_path)
            for doc in docs:
                doc.metadata.update({
                    "source": filename,
                    "file_path": file_path,
                    "processed_at": time.time()
                })
            return docs
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return []

    def process_multiple_pdfs(self, file_paths: List[str],
                            progress_callback: Optional[Callable] = None) -> List[Document]:
        all_docs = []
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_path = {
                executor.submit(self.process_single_pdf, path): path
                for path in file_paths
            }
            completed = 0
            total = len(file_paths)
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    docs = future.result()
                    all_docs.extend(docs)
                    completed += 1
                    if progress_callback:
                        progress = completed / total * 0.3
                        progress_callback(progress, f"Processed {completed}/{total} PDFs")
                except Exception as e:
                    logger.error(f"Error processing {path}: {e}")
        return all_docs

    def split_documents(self, docs: List[Document],
                       progress_callback: Optional[Callable] = None) -> List[Document]:
        if progress_callback:
            progress_callback(0.35, "Splitting documents into chunks...")
        all_chunks = []
        batch_size = self.config.batch_size
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            batch_chunks = self.text_splitter.split_documents(batch)
            all_chunks.extend(batch_chunks)
            if progress_callback:
                progress = 0.35 + (i / len(docs)) * 0.15
                progress_callback(progress, f"Split {i + len(batch)}/{len(docs)} documents")
        return all_chunks


class VectorStoreManager:
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.embedding_manager = EmbeddingManager(config)
        self.pdf_processor = FastPDFProcessor(config)

    def create_vectorstore(self, file_paths: List[str],
                          progress_callback: Optional[Callable] = None) -> int:
        try:
            if progress_callback:
                progress_callback(0.1, "Starting PDF processing...")

            all_docs = self.pdf_processor.process_multiple_pdfs(file_paths, progress_callback)

            if not all_docs:
                raise ValueError("No documents were processed successfully")

            chunks = self.pdf_processor.split_documents(all_docs, progress_callback)

            if progress_callback:
                progress_callback(0.5, f"Created {len(chunks)} chunks")

            if progress_callback:
                progress_callback(0.6, "Generating embeddings...")

            vectordb = FAISS.from_documents(chunks, self.embedding_manager.model)

            if progress_callback:
                progress_callback(0.9, "Saving vector store...")

            index_path = Path(self.config.index_dir)
            index_path.mkdir(exist_ok=True)
            vectordb.save_local(str(index_path))

            metadata = {
                "num_chunks": len(chunks),
                "num_docs": len(all_docs),
                "files": [os.path.basename(f) for f in file_paths],
                "config": self.config.__dict__,
                "created_at": time.time()
            }

            with open(index_path / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)

            if progress_callback:
                progress_callback(1.0, "Vector store created successfully!")

            return len(chunks)

        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise

    def load_vectorstore(self) -> FAISS:
        index_path = Path(self.config.index_dir)
        if not index_path.exists():
            raise FileNotFoundError(f"Vector store not found at {index_path}")
        return FAISS.load_local(
            str(index_path),
            self.embedding_manager.model,
            allow_dangerous_deserialization=True
        )

    def get_metadata(self) -> Dict[str, Any]:
        metadata_path = Path(self.config.index_dir) / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {}


class OllamaManager:
    def __init__(self, model_name: str = "mistral:latest", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            self._llm = Ollama(
                model=self.model_name,
                base_url=self.base_url,
                num_gpu=1 if torch.cuda.is_available() else 0,
                temperature=0.1
            )
        return self._llm

    def check_status(self) -> tuple:
        try:
            import requests
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m["name"] for m in models]
                if self.model_name in model_names:
                    return True, f"{self.model_name} is available"
                else:
                    return False, f"{self.model_name} not found. Available: {model_names}"
            return False, f"Ollama API returned status {response.status_code}"
        except Exception as e:
            return False, f"Cannot connect to Ollama: {str(e)}"


class RAGChain:
    def __init__(self, config: ProcessingConfig, model_name: str = "mistral"):
        self.config = config
        self.vector_manager = VectorStoreManager(config)
        self.ollama_manager = OllamaManager(model_name)
        self._chain = None
        self._retriever = None

    def initialize(self):
        try:
            vectordb = self.vector_manager.load_vectorstore()
            self._retriever = vectordb.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": 5, "score_threshold": 0.5}
            )

            prompt_template = """You are a helpful AI assistant. Use ONLY the following context to answer the question.
If the answer cannot be found in the context, say "I don't have enough information to answer that question."

Context:
{context}

Question: {question}

Provide a clear, concise answer based only on the context above:"""

            self._prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )

            self._chain = (
                RunnablePassthrough()
                | self._prompt
                | self.ollama_manager.llm
                | StrOutputParser()
            )

            return True

        except Exception as e:
            logger.error(f"Error initializing RAG chain: {e}")
            raise

    def query(self, question: str) -> Dict[str, Any]:
        if not self._chain:
            raise ValueError("RAG chain not initialized. Call initialize() first.")

        try:
            docs = self._retriever.invoke(question)
            context = "\n\n".join(doc.page_content for doc in docs)
            answer = self._chain.invoke({"context": context, "question": question})

            scores = [doc.metadata.get("score", 0.5) for doc in docs]
            avg_confidence = sum(scores) / len(scores) if scores else 0.0

            return {
                "result": answer,
                "source_documents": docs,
                "confidence": avg_confidence,
                "num_sources": len(docs)
            }

        except Exception as e:
            logger.error(f"Error querying RAG chain: {e}")
            return {
                "result": f"Error processing query: {str(e)}",
                "source_documents": [],
                "confidence": 0.0
            }


def create_rag_system(file_paths: List[str], config: ProcessingConfig = None,
                     progress_callback: Optional[Callable] = None) -> RAGChain:
    if config is None:
        config = ProcessingConfig()
    vector_manager = VectorStoreManager(config)
    vector_manager.create_vectorstore(file_paths, progress_callback)
    rag_chain = RAGChain(config)
    rag_chain.initialize()
    return rag_chain


def load_existing_rag_system(config: ProcessingConfig = None) -> RAGChain:
    if config is None:
        config = ProcessingConfig()
    rag_chain = RAGChain(config)
    rag_chain.initialize()
    return rag_chain


__all__ = [
    'GPUManager',
    'ProcessingConfig',
    'RAGChain',
    'create_rag_system',
    'load_existing_rag_system',
    'OllamaManager'
]
