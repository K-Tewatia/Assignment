# src/config.py
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for managing API keys and settings"""
    
    @staticmethod
    def get_pinecone_api_key() -> str:
        """Get Pinecone API key with validation"""
        api_key = os.getenv('PINECONE_API_KEY')
        if not api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables. Please set it in .env file")
        return api_key
    
    @staticmethod
    def get_mistral_api_key() -> str:
        """Get Mistral API key with validation"""
        api_key = os.getenv('MISTRAL_API_KEY')
        if not api_key:
            raise ValueError("MISTRAL_API_KEY not found in environment variables. Please set it in .env file")
        return api_key
    
    @staticmethod
    def get_pinecone_index_name() -> str:
        """Get Pinecone index name"""
        return os.getenv('PINECONE_INDEX_NAME', 'marketing-roi-optimizer')
    
    @staticmethod
    def get_mistral_model() -> str:
        """Get Mistral model name"""
        return os.getenv('MISTRAL_MODEL', 'mistral-large-latest')
    
    @staticmethod
    def validate_config() -> tuple[bool, list[str]]:
        """Validate all required configuration"""
        errors = []
        
        if not os.getenv('PINECONE_API_KEY'):
            errors.append("PINECONE_API_KEY is missing")
        
        if not os.getenv('MISTRAL_API_KEY'):
            errors.append("MISTRAL_API_KEY is missing")
        
        return len(errors) == 0, errors
