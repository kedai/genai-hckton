#!/usr/bin/env python3
import os
from pathlib import Path
import stanza
from typing import Optional, Tuple
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_databricks import DatabricksVectorSearch
from databricks.sdk import WorkspaceClient
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directory Configuration
HTML_DIRECTORY = "html"
INDEX_DIRECTORY = "index"

# Databricks Configuration
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")
DATABRICKS_CATALOG = os.getenv("DATABRICKS_CATALOG", "workspace_us")
DATABRICKS_SCHEMA = os.getenv("DATABRICKS_SCHEMA", "default")
VECTOR_SEARCH_INDEX_NAME = f"{DATABRICKS_CATALOG}.{DATABRICKS_SCHEMA}.doc_index"
VECTOR_SEARCH_ENDPOINT_NAME = "test-vectorstore"
EMBEDDING_DIMENSION = 768

# Model Configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "llama3.2"

class ConfigurationError(Exception):
    """Custom exception for configuration errors"""
    pass

def validate_databricks_config() -> Tuple[bool, Optional[str]]:
    """Validate Databricks configuration settings."""
    missing_vars = []
    if not DATABRICKS_HOST:
        missing_vars.append("DATABRICKS_HOST")
    if not DATABRICKS_TOKEN:
        missing_vars.append("DATABRICKS_TOKEN")

    if missing_vars:
        return False, f"Missing environment variables: {', '.join(missing_vars)}"
    return True, None

def init_stanza():
    """Initialize Stanza NLP pipeline."""
    try:
        model_dir = Path.home() / "stanza_resources" / "en"
        if not model_dir.exists():
            stanza.download('en')
        return stanza.Pipeline('en', processors='tokenize')
    except Exception as e:
        logger.error(f"Failed to initialize Stanza: {e}")
        return None

def init_databricks_client() -> Optional[WorkspaceClient]:
    """Initialize Databricks workspace client with explicit authentication."""
    is_valid, error_message = validate_databricks_config()
    if not is_valid:
        logger.error(f"Databricks configuration error: {error_message}")
        st.error(f"Databricks configuration error: {error_message}")
        st.error("Please set the required environment variables:")
        st.code("""
        export DATABRICKS_HOST="your-workspace-url"
        export DATABRICKS_TOKEN="your-access-token"
        """)
        return None

    try:
        client = WorkspaceClient(
            host=DATABRICKS_HOST,
            token=DATABRICKS_TOKEN
        )
        # Test the connection
        client.current_user.me()
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Databricks client: {e}")
        st.error(f"Failed to initialize Databricks client: {str(e)}")
        st.error("Please verify your Databricks credentials and connectivity.")
        return None

def create_vector_store(embeddings) -> Optional[DatabricksVectorSearch]:
    """Create a new vector store if it doesn't exist."""
    try:
        vector_store = DatabricksVectorSearch(
            endpoint=VECTOR_SEARCH_ENDPOINT_NAME,
            index_name=VECTOR_SEARCH_INDEX_NAME,
            embedding=embeddings,
            text_column="text"
        )

        # Test the vector store
        try:
            vector_store.as_retriever().get_relevant_documents("test")
            logger.info("Vector store initialized and tested successfully")
            return vector_store
        except Exception as e:
            logger.warning(f"Vector store test failed: {e}")
            # Even if test fails, return the store as it might be in PROVISIONING state
            return vector_store

    except Exception as e:
        logger.error(f"Failed to create vector store: {e}")
        return None

def init_vector_store(embeddings) -> Optional[DatabricksVectorSearch]:
    """Initialize Databricks Vector Search store with error handling."""
    try:
        return create_vector_store(embeddings)
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {e}")
        st.error(f"Failed to initialize Vector Search: {str(e)}")
        return None

def init_models():
    """Initialize all required models and clients with error handling."""
    try:
        # Initialize base models
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        llm = ChatOllama(model=LLM_MODEL_NAME)
        nlp = init_stanza()

        # Initialize Databricks components with validation
        workspace_client = init_databricks_client()
        if workspace_client:
            vector_store = init_vector_store(embeddings)
            if not vector_store:
                raise ConfigurationError("Failed to initialize vector store")
        else:
            raise ConfigurationError("Failed to initialize Databricks client")

        return embeddings, llm, nlp, workspace_client, vector_store

    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        st.error(f"Configuration error: {str(e)}")
        return None, None, None, None, None
    except Exception as e:
        logger.error(f"Unexpected error during initialization: {e}")
        st.error(f"Unexpected error during initialization: {str(e)}")
        return None, None, None, None, None

# Global initialization with error handling
try:
    embeddings, llm, nlp, workspace_client, vector_store = init_models()
    if not all([embeddings, llm, nlp, workspace_client, vector_store]):
        st.warning("Some components failed to initialize. The application may have limited functionality.")
        logger.warning("Some components failed to initialize")
except Exception as e:
    logger.error(f"Error during initialization: {e}")
    st.error(f"Error during initialization: {str(e)}")
    st.error("The application may not function correctly. Please check the configuration.")
    # Provide fallback values to allow partial functionality
    embeddings, llm, nlp = None, None, None
    workspace_client, vector_store = None, None