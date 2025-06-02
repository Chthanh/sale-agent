"""
Configuration module for the Agentic RAG system
"""
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    """Configuration settings for the RAG system"""
    
    # Model configurations
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL_NAME: str = "gemini-2.0-flash"
    LLM_TEMPERATURE: float = 0.0
    
    # ChromaDB configurations
    CHROMA_COLLECTION_NAME: str = "product_catalog"
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    
    # Search configurations
    WEB_SEARCH_RESULTS: int = 3
    MAX_RETRIEVAL_DOCS: int = 5
    
    # API Keys (set via environment variables)
    GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")
    TAVILY_API_KEY: Optional[str] = os.getenv("TAVILY_API_KEY")
    
    def validate(self) -> bool:
        """Validate configuration settings"""
        if not self.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
        if not self.TAVILY_API_KEY:
            raise ValueError("TAVILY_API_KEY environment variable is required")
        return True

# Global configuration instance
config = Config()