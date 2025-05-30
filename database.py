"""
Vector database handler for the RAG system
"""
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List, Optional
import logging

from config import config

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename='app.log',
    filemode='a',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)


class VectorDatabase:
    """Handles vector database operations"""
    
    def __init__(self):
        self.embedding_model = None
        self.chroma_db = None
        self.retriever = None
        self._initialize()
    
    def _initialize(self):
        """Initialize the vector database and embeddings"""
        try:
            # Initialize embedding model
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=config.EMBEDDING_MODEL_NAME
            )
            
            # Load ChromaDB
            self.chroma_db = Chroma(
                collection_name=config.CHROMA_COLLECTION_NAME,
                embedding_function=self.embedding_model,
                persist_directory=config.CHROMA_PERSIST_DIR
            )
            
            # Create retriever
            self.retriever = self.chroma_db.as_retriever(
                search_kwargs={"k": config.MAX_RETRIEVAL_DOCS}
            )
            
            logging.info("Vector database initialized successfully")
            
        except Exception as e:
            logging.error(f"Failed to initialize vector database: {e}")
            raise
    
    def retrieve_documents(self, query: str) -> List[Document]:
        """Retrieve relevant documents for a query"""
        try:
            documents = self.retriever.invoke(query)
            logging.info(f"Retrieved {len(documents)} documents for query: {query}")
            return documents
        except Exception as e:
            logging.error(f"Error retrieving documents: {e}")
            return []
    
    def add_documents(self, documents: List[Document]) -> bool:
        """Add new documents to the vector database""" 
        try:
            self.chroma_db.add_documents(documents)
            logging.info(f"Added {len(documents)} documents to database")
            return True
        except Exception as e:
            logging.error(f"Error adding documents: {e}")
            return False
    
    def search_similarity(self, query: str, k: int = 5) -> List[tuple]:
        """Search for similar documents with scores"""
        try:
            results = self.chroma_db.similarity_search_with_score(query, k=k)
            return results
        except Exception as e:
            logging.error(f"Error in similarity search: {e}")
            return []


# Global database instance
vector_db = VectorDatabase()
# print(vector_db.retrieve_documents("samsung"))