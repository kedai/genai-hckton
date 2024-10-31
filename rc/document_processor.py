#!/usr/bin/env python3
import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import requests
import pandas as pd
from uuid import uuid4
import time
import streamlit as st
from langchain.schema import Document
from langchain_databricks import DatabricksVectorSearch
from dp_html_parser import HTMLContentParser
import time

from config import (
    DATABRICKS_HOST,
    DATABRICKS_TOKEN,
    DATABRICKS_CATALOG,
    DATABRICKS_SCHEMA,
    VECTOR_SEARCH_INDEX_NAME,
    VECTOR_SEARCH_ENDPOINT_NAME,
    EMBEDDING_DIMENSION,
    workspace_client,
    nlp
)

class DocumentProcessor:
    def __init__(
        self,
        html_directory: str,
        index_directory: str,
        embeddings,
        llm
    ):
        # Configure logging first
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Basic setup
        self.html_directory = Path(html_directory)
        self.index_directory = Path(index_directory)
        self.llm = llm.model
        self.embeddings = embeddings

        # Initialize vector store client
        self.vector_store = self._initialize_vector_store()

        # Processing configuration
        self.batch_size = 50

        # Set up caching infrastructure
        self.cache_directory = self.index_directory / "cache"
        self.cache_directory.mkdir(parents=True, exist_ok=True)
        self.processed_cache_file = self.cache_directory / "processed_files.json"
        self.vectorstore_info_file = self.cache_directory / "vectorstore_info.json"

        # External services configuration
        self.ollama_url = "http://localhost:11434/api/generate"

        # Initialize status tracking
        self._init_status()

    def _init_status(self):
        """Initialize processing status tracking."""
        if 'processing_status' not in st.session_state:
            st.session_state.processing_status = {
                'current_file': '',
                'total_files': 0,
                'processed_files': 0,
                'total_documents': 0,
                'current_batch': 0,
                'start_time': None,
                'errors': []
            }
    def _initialize_vector_store(self) -> Optional[DatabricksVectorSearch]:
        """
        Initialize and validate the Databricks Vector Search index.
        Attempts to connect to existing index before creating new one.

        Returns:
            Optional[DatabricksVectorSearch]: Initialized vector store or None if initialization fails
        """
        try:
            self.logger.info("Initializing vector store connection...")

            # Initialize vector store connection
            vector_store = DatabricksVectorSearch(
                endpoint=VECTOR_SEARCH_ENDPOINT_NAME,
                index_name=VECTOR_SEARCH_INDEX_NAME,
                embedding=self.embeddings,
                text_column="text"
            )

            # Test if the index exists and is queryable
            try:
                # Perform a simple test query
                vector_store.similarity_search("test", k=1)
                self.logger.info("Successfully connected to existing vector store")
                return vector_store
            except Exception as e:
                if "does not exist" in str(e).lower():
                    self.logger.info("Vector store index does not exist, will create new one")
                    return self._create_new_vector_store()
                elif "PROVISIONING" in str(e).upper():
                    self.logger.info("Vector store index is in PROVISIONING state")
                    return vector_store
                else:
                    self.logger.warning(f"Vector store test query failed: {e}")
                    return vector_store

        except Exception as e:
            self.logger.error(f"Error initializing vector store: {e}")
            return None

    def _create_new_vector_store(self) -> Optional[DatabricksVectorSearch]:
        """
        Create a new vector store index in Databricks.

        Returns:
            Optional[DatabricksVectorSearch]: Newly created vector store or None if creation fails
        """
        try:
            self.logger.info("Creating new vector store index...")

            vector_store = DatabricksVectorSearch(
                endpoint=VECTOR_SEARCH_ENDPOINT_NAME,
                index_name=VECTOR_SEARCH_INDEX_NAME,
                embedding=self.embeddings,
                text_column="text"
            )

            # Create the index with specific settings
            vector_store.create_index(
                primary_key="id",
                embedding_dimension=EMBEDDING_DIMENSION,
                sync_mode="SYNC"  # Use synchronous mode for better reliability
            )

            self.logger.info("Created new vector store index successfully")
            return vector_store
        except Exception as e:
            self.logger.error(f"Failed to create new vector store: {e}")
            return None

    def _load_cache_info(self) -> Dict:
        """
        Load cached information about processed files and vectorstore state.
        Enhanced with better error handling and validation.

        Returns:
            Dict: Cache information including processed files and vectorstore state
        """
        cache_info = {
            "processed_files": {},
            "vectorstore_hash": None,
            "last_modified": None
        }

        try:
            if self.processed_cache_file.exists():
                with open(self.processed_cache_file, 'r') as f:
                    cache_data = json.load(f)
                    # Validate cache data structure
                    if isinstance(cache_data, dict):
                        cache_info["processed_files"] = cache_data

            if self.vectorstore_info_file.exists():
                with open(self.vectorstore_info_file, 'r') as f:
                    vectorstore_info = json.load(f)
                    if isinstance(vectorstore_info, dict):
                        if "vectorstore_hash" in vectorstore_info:
                            cache_info["vectorstore_hash"] = vectorstore_info["vectorstore_hash"]
                        if "last_modified" in vectorstore_info:
                            cache_info["last_modified"] = vectorstore_info["last_modified"]

        except Exception as e:
            self.logger.error(f"Error loading cache info: {e}")
            # Invalidate cache if there's an error
            cache_info = {
                "processed_files": {},
                "vectorstore_hash": None,
                "last_modified": None
            }

        return cache_info

    def _save_cache_info(
        self,
        processed_files: Dict,
        vectorstore_hash: Optional[str] = None
    ) -> None:
        """
        Save cache information about processed files and vectorstore state.
        Enhanced with atomic writes for better reliability.

        Args:
            processed_files (Dict): Information about processed files
            vectorstore_hash (Optional[str]): Hash of the vectorstore state
        """
        try:
            # Save processed files info
            temp_processed_file = self.processed_cache_file.with_suffix('.tmp')
            with open(temp_processed_file, 'w') as f:
                json.dump(processed_files, f)
            temp_processed_file.replace(self.processed_cache_file)

            # Save vectorstore info if provided
            if vectorstore_hash is not None:
                vectorstore_info = {
                    "vectorstore_hash": vectorstore_hash,
                    "last_modified": datetime.now().isoformat()
                }
                temp_vectorstore_file = self.vectorstore_info_file.with_suffix('.tmp')
                with open(temp_vectorstore_file, 'w') as f:
                    json.dump(vectorstore_info, f)
                temp_vectorstore_file.replace(self.vectorstore_info_file)

        except Exception as e:
            self.logger.error(f"Error saving cache info: {e}")
            # Try to clean up temporary files if they exist
            try:
                if temp_processed_file.exists():
                    temp_processed_file.unlink()
                if temp_vectorstore_file.exists():
                    temp_vectorstore_file.unlink()
            except:
                pass

    def _check_vectorstore_validity(self) -> bool:
        """
        Enhanced vector store validation that checks both existence and content.
        Includes additional checks for index health and content consistency.

        Returns:
            bool: True if vector store is valid and up-to-date, False otherwise
        """
        try:
            if self.vector_store is None:
                return False

            # Load cache information
            cache_info = self._load_cache_info()

            # Check if we have processed files info
            if not cache_info.get("processed_files"):
                return False

            # Compare current files with cached files
            current_files = {f for f in os.listdir(self.html_directory) if f.endswith('.html')}
            cached_files = set(cache_info["processed_files"].keys())

            # Check for new or modified files
            if current_files - cached_files:
                return False

            for file_name in current_files:
                file_path = self.html_directory / file_name
                current_modified = os.path.getmtime(file_path)
                cached_modified = cache_info["processed_files"].get(file_name)

                if not cached_modified or current_modified > cached_modified:
                    return False

            # Verify vector store is queryable with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.vector_store.similarity_search("test", k=1)
                    return True
                except Exception as e:
                    if "PROVISIONING" in str(e).upper():
                        time.sleep(5)  # Wait for index to be ready
                        continue
                    elif attempt == max_retries - 1:
                        self.logger.warning(f"Vector store validation failed after {max_retries} attempts: {e}")
                        return False
                    time.sleep(1 * (attempt + 1))

            return False

        except Exception as e:
            self.logger.error(f"Error checking vectorstore validity: {e}")
            return False

    def _process_html(self, html_content: str, file_name: str) -> List[Document]:
        """
        Process HTML content into documents with enhanced table analysis and content chunking.

        Args:
            html_content (str): Raw HTML content to process
            file_name (str): Name of the source file

        Returns:
            List[Document]: List of processed documents
        """
        documents = []
        parser = HTMLContentParser(html_content)

        # Process main content with improved chunking
        contents = parser.get_contents()
        if contents:
            # Group related content together
            current_chunk = []
            current_chunk_size = 0
            max_chunk_size = 500  # Characters per chunk

            for content in contents:
                content_size = len(content)

                # Check if adding this content would exceed chunk size
                if current_chunk_size + content_size > max_chunk_size and current_chunk:
                    # Create document from current chunk
                    chunk_text = '\n'.join(current_chunk)
                    self.logger.debug(f"Created chunk of size {len(chunk_text)} from {file_name}")

                    documents.append(Document(
                        page_content=chunk_text,
                        metadata={
                            "source": file_name,
                            "type": "main_content",
                            "chunk_size": len(chunk_text)
                        }
                    ))
                    current_chunk = []
                    current_chunk_size = 0

                current_chunk.append(content)
                current_chunk_size += content_size

            # Handle remaining chunk
            if current_chunk:
                chunk_text = '\n'.join(current_chunk)
                documents.append(Document(
                    page_content=chunk_text,
                    metadata={
                        "source": file_name,
                        "type": "main_content",
                        "chunk_size": len(chunk_text)
                    }
                ))

        # Process tables with enhanced context and analysis
        tables = parser.get_tables()
        for idx, table in enumerate(tables):
            try:
                # Find most relevant context for this table
                context = self._find_table_context(table, contents)

                # Attempt table analysis with retries and backoff
                max_retries = 3
                table_analysis = None

                for attempt in range(max_retries):
                    try:
                        table_analysis = self._analyze_table_with_ollama(table, context)
                        if table_analysis:
                            break
                    except Exception as e:
                        if attempt == max_retries - 1:
                            self.logger.warning(f"Failed to analyze table after {max_retries} attempts: {e}")
                        time.sleep(1 * (attempt + 1))  # Exponential backoff

                if not table_analysis:
                    table_analysis = self._format_table(table)

                table_content = f"""
                Table Analysis:
                {table_analysis}

                Raw Table Data:
                {self._format_table(table)}

                Related Context:
                {context}
                """.strip()

                documents.append(Document(
                    page_content=table_content,
                    metadata={
                        "source": file_name,
                        "type": "table",
                        "table_index": idx,
                        "has_analysis": bool(table_analysis)
                    }
                ))
            except Exception as e:
                self.logger.warning(f"Error processing table {idx} in {file_name}: {e}")
                # Fallback to basic table formatting
                documents.append(Document(
                    page_content=self._format_table(table),
                    metadata={
                        "source": file_name,
                        "type": "table",
                        "table_index": idx,
                        "has_analysis": False
                    }
                ))

        # Process images with enhanced context
        images = parser.get_images()
        for idx, image in enumerate(images):
            if image.get('alt', '').strip():
                # Find relevant context for the image
                image_context = self._get_image_context(image, contents)
                image_content = f"""
                Image Content:
                {image['alt']}

                Related Context:
                {image_context}
                """.strip()

                documents.append(Document(
                    page_content=image_content,
                    metadata={
                        "source": file_name,
                        "type": "image",
                        "image_index": idx
                    }
                ))

        return documents

    def _format_table(self, table: Dict) -> str:
        """
        Format table data in a structured string representation.

        Args:
            table (Dict): Dictionary containing table headers and rows

        Returns:
            str: Formatted table string
        """
        if not table.get('headers') or not table.get('rows'):
            return ""

        formatted = "Table Structure:\n"
        headers = table['headers']
        rows = table['rows']

        # Calculate column widths
        col_widths = [len(str(h)) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))

        # Format headers
        header_row = " | ".join(f"{str(header):{width}}" for header, width in zip(headers, col_widths))
        formatted += header_row + "\n"
        formatted += "-" * len(header_row) + "\n"

        # Format rows
        for row in rows:
            formatted += " | ".join(f"{str(cell):{width}}" for cell, width in zip(row, col_widths)) + "\n"

        # Add summary
        formatted += f"\nSummary: Table with {len(headers)} columns and {len(rows)} rows."

        return formatted

    def _analyze_table_with_ollama(self, table: Dict, context: str = "") -> str:
        """
        Analyze table content using Ollama with improved error handling and retry logic.

        Args:
            table (Dict): Dictionary containing table data
            context (str): Additional context for table analysis

        Returns:
            str: Analysis of the table content
        """
        try:
            headers = table['headers']
            rows = table['rows'][:5]  # Limit to first 5 rows for analysis

            # Determine appropriate analysis approach
            if any('status' in h.lower() for h in headers):
                analysis_type = "status/state table"
                focus_points = [
                    "state transitions or status codes",
                    "relationships between columns",
                    "conditions or requirements"
                ]
            elif any('parameter' in h.lower() or 'field' in h.lower() for h in headers):
                analysis_type = "parameter/field table"
                focus_points = [
                    "key parameters/fields and their purposes",
                    "required vs optional fields",
                    "data type requirements or constraints"
                ]
            else:
                analysis_type = "documentation table"
                focus_points = [
                    "main purpose of this table",
                    "relationships between columns",
                    "patterns or requirements"
                ]

            prompt = f"""
            Analyze this API {analysis_type}:
            Headers: {', '.join(headers)}
            Sample Data: {str(rows)}
            Context: {context}

            Please provide a concise analysis focusing on:
            {' - '.join(focus_points)}
            """

            # Get model name and make request with retry logic
            model_name = getattr(self.llm, 'model', 'llama3.2')
            max_retries = 3

            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        self.ollama_url,
                        headers={"Content-Type": "application/json"},
                        json={
                            "model": model_name,
                            "prompt": prompt,
                            "stream": False
                        },
                        timeout=30
                    )

                    if response.status_code == 200:
                        result = response.json()
                        analysis = result.get('response', '').strip()
                        return analysis if analysis else self._format_table(table)

                    if attempt < max_retries - 1:
                        time.sleep(1 * (attempt + 1))
                        continue

                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(1 * (attempt + 1))
                        continue
                    self.logger.warning(f"Ollama request failed: {e}")

            return self._format_table(table)

        except Exception as e:
            self.logger.warning(f"Error analyzing table: {e}")
            return self._format_table(table)

    def _find_table_context(self, table: Dict, contents: List[str]) -> str:
        """
        Find relevant context for a table from surrounding content with improved matching.

        Args:
            table (Dict): The table to find context for
            contents (List[str]): List of content strings to search through

        Returns:
            str: Relevant context for the table
        """
        if not contents or not table.get('headers'):
            return ""

        # Extract key terms from table headers and first row
        key_terms = set()

        # Process headers
        for header in table['headers']:
            # Split camelCase and snake_case terms
            terms = header.replace('_', ' ').split()
            for term in terms:
                # Split camelCase
                camel_terms = ''.join(' ' + char if char.isupper() else char for char in term).strip().split()
                key_terms.update(word.lower() for word in camel_terms)

        # Process first row if available
        if table.get('rows') and table['rows'][0]:
            for cell in table['rows'][0]:
                if isinstance(cell, str):
                    terms = cell.replace('_', ' ').split()
                    key_terms.update(word.lower() for word in terms)

        # Find paragraphs with matching terms
        related_paragraphs = []
        for content in contents:
            # Normalize content terms
            content_words = set(word.lower() for word in content.replace('_', ' ').split())

            # Calculate relevance score
            matching_terms = len(key_terms & content_words)
            if matching_terms >= 2:  # Require at least 2 matching terms
                # Calculate additional relevance factors
                term_density = matching_terms / len(content_words)
                proximity_bonus = 1.0  # Could be adjusted based on distance from table

                # Combined relevance score
                relevance_score = matching_terms * term_density * proximity_bonus

                related_paragraphs.append((relevance_score, content))

        # Sort by relevance and combine most relevant contexts
        related_paragraphs.sort(reverse=True)
        selected_contexts = []
        total_length = 0
        max_context_length = 500

        for _, content in related_paragraphs:
            if total_length + len(content) <= max_context_length:
                selected_contexts.append(content)
                total_length += len(content)
            else:
                break

        context = " ".join(selected_contexts)
        if len(context) > max_context_length:
            context = context[:max_context_length-3] + "..."

        return context

    def _get_image_context(self, image: Dict, contents: List[str]) -> str:
        """
        Get surrounding text context for an image with improved relevance matching.

        Args:
            image (Dict): Image information including alt text
            contents (List[str]): List of content strings to search through

        Returns:
            str: Context relevant to the image
        """
        if not image.get('alt') or not contents:
            return ""

        # Extract key terms from alt text
        alt_text = image['alt'].lower()
        alt_words = set(alt_text.split())

        # Score each content block based on relevance
        scored_contents = []
        for content in contents:
            content_lower = content.lower()

            # Calculate various relevance factors
            term_matches = sum(1 for word in alt_words if word in content_lower)
            exact_phrase_bonus = 2 if alt_text in content_lower else 0
            content_length_penalty = 1 / (1 + len(content) / 500)  # Prefer shorter, relevant contexts

            # Combined relevance score
            relevance_score = (term_matches + exact_phrase_bonus) * content_length_penalty

            if relevance_score > 0:
                scored_contents.append((relevance_score, content))

        # Sort by relevance score and select best context
        if scored_contents:
            scored_contents.sort(reverse=True)
            context = scored_contents[0][1]

            # Trim context if too long
            max_length = 300
            if len(context) > max_length:
                context = context[:max_length-3] + "..."

            return context

        return ""

    def _process_documents(self, documents: List[Document]) -> List[Document]:
        """
        Process documents into optimal chunks with improved handling of different content types.

        Args:
            documents (List[Document]): List of documents to process

        Returns:
            List[Document]: Processed and chunked documents
        """
        processed_docs = []

        for doc in documents:
            if not doc.page_content.strip():
                continue

            try:
                # Handle different content types appropriately
                if doc.metadata.get('type') in ['table', 'image']:
                    # Don't split structured content
                    processed_docs.append(doc)
                    continue

                # Process main content into sentences with context preservation
                nlp_doc = nlp(doc.page_content)

                current_chunk = []
                current_chunk_size = 0
                max_chunk_size = 500

                current_section = None
                section_buffer = []

                for sentence in nlp_doc.sentences:
                    sentence_text = sentence.text.strip()
                    if not sentence_text:
                        continue

                    sentence_size = len(sentence_text)

                    # Check for section headers or significant content breaks
                    if self._is_section_header(sentence_text):
                        # Process existing section if any
                        if section_buffer:
                            processed_docs.extend(self._process_section(
                                section_buffer,
                                current_section,
                                doc.metadata
                            ))
                        current_section = sentence_text
                        section_buffer = []
                        continue

                    section_buffer.append(sentence_text)

                    # Process buffer if it gets too large
                    if sum(len(s) for s in section_buffer) > max_chunk_size:
                        processed_docs.extend(self._process_section(
                            section_buffer,
                            current_section,
                            doc.metadata
                        ))
                        section_buffer = []

                # Process any remaining content
                if section_buffer:
                    processed_docs.extend(self._process_section(
                        section_buffer,
                        current_section,
                        doc.metadata
                    ))

            except Exception as e:
                self.logger.error(f"Error processing document: {e}")
                if doc.page_content.strip():
                    processed_docs.append(doc)

        return processed_docs

    def _is_section_header(self, text: str) -> bool:
        """
        Determine if a piece of text is likely a section header.

        Args:
            text (str): Text to analyze

        Returns:
            bool: True if text appears to be a section header
        """
        # Common header indicators
        if len(text.split()) <= 5 and any([
            text.endswith(':'),
            text.isupper(),
            text.istitle() and len(text) < 50,
            any(text.startswith(prefix) for prefix in ['Chapter ', 'Section ', '#']),
            text.strip().split()[0] in ['Overview', 'Introduction', 'Summary', 'Details', 'Example', 'Notes']
        ]):
            return True
        return False

    def _process_section(self, sentences: List[str], section_header: Optional[str], metadata: Dict) -> List[Document]:
        """
        Process a section of content into appropriately sized chunks.

        Args:
            sentences (List[str]): List of sentences in the section
            section_header (Optional[str]): Section header if any
            metadata (Dict): Document metadata

        Returns:
            List[Document]: Processed document chunks
        """
        documents = []
        current_chunk = []
        current_chunk_size = 0
        max_chunk_size = 500

        # Add section header if present
        if section_header:
            current_chunk.append(section_header)
            current_chunk_size = len(section_header)

        for sentence in sentences:
            sentence_size = len(sentence)

            if current_chunk_size + sentence_size > max_chunk_size and current_chunk:
                # Create document from current chunk
                chunk_text = ' '.join(current_chunk)
                documents.append(Document(
                    page_content=chunk_text,
                    metadata={
                        **metadata,
                        'chunk_size': len(chunk_text),
                        'has_section_header': bool(section_header)
                    }
                ))
                current_chunk = []
                current_chunk_size = 0

            current_chunk.append(sentence)
            current_chunk_size += sentence_size

        # Handle remaining chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            documents.append(Document(
                page_content=chunk_text,
                metadata={
                    **metadata,
                    'chunk_size': len(chunk_text),
                    'has_section_header': bool(section_header)
                }
            ))

        return documents
    def _add_documents_batch(self, documents: List[Dict]) -> None:
        """
        Add a batch of documents to the vector store with improved error handling and retries.

        Args:
            documents (List[Dict]): List of document dictionaries to add
        """
        if not documents:
            return

        try:
            # Update processing status
            st.session_state.processing_status['current_batch'] += 1

            # Generate embeddings for the batch
            texts = [doc['text'] for doc in documents]

            # Retry logic for embedding generation
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    embeddings = self.embeddings.embed_documents(texts)
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(1 * (attempt + 1))

            # Create records with the correct schema
            records = []
            for doc, embedding in zip(documents, embeddings):
                record = {
                    "id": str(uuid4()),
                    "text": doc['text'],
                    "embedding": embedding,
                    "source": doc['source'],
                    "type": doc.get('type', 'content'),
                }
                records.append(record)

            # Convert to DataFrame for batch processing
            df = pd.DataFrame(records)

            # Attempt to add to vector store with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.vector_store.add_texts(
                        texts=texts,
                        metadatas=[{
                            'id': record['id'],
                            'source': record['source'],
                            'type': record['type']
                        } for record in records]
                    )
                    self.logger.info(f"Successfully added batch of {len(records)} documents")
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(1 * (attempt + 1))

            # Update processing status
            st.session_state.processing_status['total_documents'] += len(records)

        except Exception as e:
            error_msg = f"Error adding batch to vector store: {e}"
            self.logger.error(error_msg)
            st.session_state.processing_status['errors'].append(error_msg)
            raise

    def get_or_create_vectorstore(self) -> Optional[DatabricksVectorSearch]:
        """
        Get existing vectorstore if valid, otherwise create new one.
        Enhanced with better status reporting and error handling.

        Returns:
            Optional[DatabricksVectorSearch]: The vectorstore instance or None if creation fails
        """
        try:
            # First check for valid existing vector store
            if self.vector_store and self._check_vectorstore_validity():
                self.logger.info("Using existing valid vector store")
                return self.vector_store

            # If vector store is invalid or doesn't exist, try to initialize it
            if not self.vector_store:
                self.vector_store = self._initialize_vector_store()

            # If we still don't have a valid vector store, we need to ingest
            if not self._check_vectorstore_validity():
                self.logger.info("Need to process HTML files")
                vectorstore, _ = self.ingest_html()
                return vectorstore

            return self.vector_store

        except Exception as e:
            self.logger.error(f"Error in get_or_create_vectorstore: {e}")
            return None

    def add_documents(self, documents: List[Document]) -> Optional[Tuple[DatabricksVectorSearch, int]]:
        """
        Add new documents to the existing vectorstore with improved error handling.

        Args:
            documents (List[Document]): List of documents to add

        Returns:
            Optional[Tuple[DatabricksVectorSearch, int]]: Tuple of (vectorstore, document_count) or None if addition fails
        """
        try:
            processed_docs = self._process_documents(documents)
            doc_dicts = []

            for doc in processed_docs:
                text_content = doc.page_content.strip()
                if text_content:
                    doc_data = {
                        "text": text_content,
                        "source": doc.metadata.get("source", ""),
                        "type": doc.metadata.get("type", "content"),
                    }
                    doc_dicts.append(doc_data)

            if doc_dicts:
                # Process in batches
                batch_size = self.batch_size
                for i in range(0, len(doc_dicts), batch_size):
                    batch = doc_dicts[i:i + batch_size]
                    self._add_documents_batch(batch)

            return self.vector_store, len(doc_dicts)

        except Exception as e:
            self.logger.error(f"Error adding documents: {e}")
            return None, 0

    def ingest_html(self) -> Optional[Tuple[DatabricksVectorSearch, int]]:
        """
        Ingest HTML files into Databricks Vector Search with improved progress tracking and error handling.

        Returns:
            Optional[Tuple[DatabricksVectorSearch, int]]: Tuple of (vectorstore, document_count) or None if ingestion fails
        """
        try:
            # Initialize/reset processing status
            st.session_state.processing_status.update({
                'start_time': datetime.now(),
                'current_file': '',
                'processed_files': 0,
                'total_documents': 0,
                'current_batch': 0,
                'errors': []
            })

            # Load cache information
            cache_info = self._load_cache_info()
            processed_files = cache_info["processed_files"]
            total_processed = 0
            documents = []
            updated = False

            # Set up progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Get list of HTML files
            html_files = [f for f in os.listdir(self.html_directory) if f.endswith('.html')]
            st.session_state.processing_status['total_files'] = len(html_files)

            for idx, file_name in enumerate(html_files):
                file_path = self.html_directory / file_name
                current_modified = os.path.getmtime(file_path)

                # Skip if file hasn't changed
                if file_name in processed_files and processed_files[file_name] == current_modified:
                    continue

                try:
                    st.session_state.processing_status['current_file'] = file_name
                    status_text.text(f"Processing {file_name}...")

                    with open(file_path, 'r', encoding='utf-8') as file:
                        html_content = file.read()

                    # Process the HTML content
                    docs = self._process_html(html_content, file_name)
                    processed_docs = self._process_documents(docs)

                    # Prepare documents for vector store
                    for doc in processed_docs:
                        text_content = doc.page_content.strip()
                        if text_content:
                            doc_data = {
                                "text": text_content,
                                "source": doc.metadata.get("source", ""),
                                "type": doc.metadata.get("type", "content"),
                            }
                            documents.append(doc_data)

                        # Process in batches
                        if len(documents) >= self.batch_size:
                            self._add_documents_batch(documents)
                            documents = []

                    total_processed += len(processed_docs)
                    processed_files[file_name] = current_modified
                    updated = True

                    # Update status
                    progress = (idx + 1) / len(html_files)
                    progress_bar.progress(progress)
                    status_text.text(
                        f"Processed {total_processed} documents from "
                        f"{idx + 1}/{len(html_files)} files"
                    )
                    st.session_state.processing_status['processed_files'] = idx + 1

                except Exception as e:
                    error_msg = f"Error processing {file_name}: {str(e)}"
                    self.logger.error(error_msg)
                    st.session_state.processing_status['errors'].append(error_msg)
                    continue

            # Process remaining documents
            if documents:
                try:
                    self._add_documents_batch(documents)
                except Exception as e:
                    error_msg = f"Error processing final batch: {str(e)}"
                    self.logger.error(error_msg)
                    st.session_state.processing_status['errors'].append(error_msg)

            # Update cache if any files were processed
            if updated:
                vectorstore_hash = str(hash(frozenset(processed_files.items())))
                self._save_cache_info(processed_files, vectorstore_hash)

            # Clean up progress indicators
            progress_bar.empty()
            status_text.empty()

            # Display summary
            if st.session_state.processing_status['errors']:
                st.warning(
                    f"Completed with {len(st.session_state.processing_status['errors'])} errors. "
                    "Check the logs for details."
                )

            return self.vector_store, total_processed

        except Exception as e:
            self.logger.error(f"Error during ingestion: {e}")
            st.error(f"Error during document ingestion: {str(e)}")
            return None, 0

