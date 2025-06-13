
import os
import logging
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGRetriever:
    def __init__(self, corpus_dir="rag_corpus"):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = None
        self.load_corpus(corpus_dir)

    def load_corpus(self, corpus_dir):
        try:
            documents = []
            for filename in os.listdir(corpus_dir):
                if filename.endswith(".md"):
                    file_path = os.path.join(corpus_dir, filename)
                    loader = TextLoader(file_path, encoding="utf-8")
                    raw_docs = loader.load()
                    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=50)
                    split_docs = text_splitter.split_documents(raw_docs)
                    for doc in split_docs:
                        doc.metadata["source"] = filename
                        doc.metadata["level"] = "all"  # Corpus is general for all levels
                    documents.extend(split_docs)
            if not documents:
                raise ValueError("No documents found in corpus directory")
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            logger.info(f"Loaded {len(documents)} document chunks into FAISS vector store")
        except Exception as e:
            logger.error(f"Error loading corpus: {e}")
            raise

    def retrieve(self, query, level, n_results=2):
        try:
            query = f"{query} level:{level}"
            docs = self.vector_store.similarity_search(query, k=n_results)
            return [doc.page_content for doc in docs]
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
