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
        """
        Initialize the DocumentProcessor.

        Args:
            html_directory (str): Directory containing HTML files
            index_directory (str): Directory for storing cache files
            embeddings: Embedding model for vectorization
            llm: Language model for text analysis
        """
        # Basic setup
        self.html_directory = Path(html_directory)
        self.index_directory = Path(index_directory)
        self.llm = llm.model
        self.embeddings = embeddings

        # Initialize vector store with proper configuration
        self.vector_store = DatabricksVectorSearch(
            endpoint=VECTOR_SEARCH_ENDPOINT_NAME,
            index_name=VECTOR_SEARCH_INDEX_NAME,
            embedding=embeddings,
            text_column="text",

        )

        # Configure logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Processing configuration
        self.batch_size = 50

        # Set up caching infrastructure
        self.cache_directory = self.index_directory / "cache"
        self.cache_directory.mkdir(parents=True, exist_ok=True)
        self.processed_cache_file = self.cache_directory / "processed_files.json"
        self.vectorstore_info_file = self.cache_directory / "vectorstore_info.json"

        # External services configuration
        self.ollama_url = "http://localhost:11434/api/generate"

    def _load_cache_info(self) -> Dict:
        """
        Load cached information about processed files and vectorstore state.

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
                    cache_info["processed_files"] = json.load(f)

            if self.vectorstore_info_file.exists():
                with open(self.vectorstore_info_file, 'r') as f:
                    vectorstore_info = json.load(f)
                    cache_info.update(vectorstore_info)

        except Exception as e:
            self.logger.error(f"Error loading cache info: {e}")

        return cache_info

    def _save_cache_info(
        self,
        processed_files: Dict,
        vectorstore_hash: Optional[str] = None
    ) -> None:
        """
        Save cache information about processed files and vectorstore state.

        Args:
            processed_files (Dict): Information about processed files
            vectorstore_hash (Optional[str]): Hash of the vectorstore state
        """
        try:
            with open(self.processed_cache_file, 'w') as f:
                json.dump(processed_files, f)

            if vectorstore_hash is not None:
                vectorstore_info = {
                    "vectorstore_hash": vectorstore_hash,
                    "last_modified": datetime.now().isoformat()
                }
                with open(self.vectorstore_info_file, 'w') as f:
                    json.dump(vectorstore_info, f)

        except Exception as e:
            self.logger.error(f"Error saving cache info: {e}")

    def _check_vectorstore_validity(self) -> bool:
        """
        Check if the existing Databricks Vector Search index is valid and up-to-date.

        Returns:
            bool: True if vectorstore is valid and up-to-date, False otherwise
        """
        try:
            # Verify vector store existence and access
            # try:
            #     # Instead of performing a search, just check if we can access the index
            #     if self.vector_store is None:
            #         return False
            #     # The index might be in PROVISIONING state, so we'll consider it valid
            #     return True
            # except Exception as e:
            #     self.logger.warning(f"Vector store access check failed: {e}")
            #     return False

            # Load and check cache information
            cache_info = self._load_cache_info()
            if not cache_info.get("vectorstore_hash"):
                return False

            # Compare current files with cached files
            current_files = {f for f in os.listdir(self.html_directory) if f.endswith('.html')}
            cached_files = set(cache_info["processed_files"].keys())

            if current_files - cached_files:
                return False

            # Check file modifications
            for file_name in current_files:
                file_path = self.html_directory / file_name
                current_modified = os.path.getmtime(file_path)
                cached_modified = cache_info["processed_files"].get(file_name)

                if not cached_modified or current_modified > cached_modified:
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Error checking vectorstore validity: {e}")
            return False

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
        formatted += f"Headers: {' | '.join(table['headers'])}\n\n"

        for row in table['rows']:
            formatted += f"Row: {' | '.join(str(cell) for cell in row)}\n"

        if len(table['rows']) > 0:
            formatted += f"\nSummary: Table with {len(table['headers'])} columns and {len(table['rows'])} rows."

        return formatted

    def _analyze_table_with_ollama(self, table: Dict, context: str = "") -> str:
        """
        Analyze table content using Ollama to extract meaning and relationships.
        Updated to properly handle model name and request format.

        Args:
            table (Dict): Dictionary containing table data
            context (str): Additional context for table analysis

        Returns:
            str: Analysis of the table content
        """
        try:
            headers = table['headers']
            rows = table['rows']  # Limit to first 5 rows for analysis to avoid overloading
            print(f"Analyzing table with headers: {headers}")

            # Determine the appropriate analysis prompt based on table content
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

            # Get the model name from the ChatOllama instance
            model_name = getattr(self.llm, 'model', 'llama3.2')
            # Attempt request with retry logic
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
                        try:
                            result = response.json()
                            if isinstance(result, dict):
                                return result.get('response', '').strip() or self._format_table(table)
                            return self._format_table(table)
                        except json.JSONDecodeError:
                            # Try parsing line by line for streaming responses
                            full_response = ""
                            for line in response.text.strip().split('\n'):
                                try:
                                    data = json.loads(line)
                                    full_response += data.get('response', '')
                                except json.JSONDecodeError:
                                    continue
                            return full_response.strip() or self._format_table(table)

                    if attempt < max_retries - 1:
                        time.sleep(1 * (attempt + 1))  # Exponential backoff
                        continue

                    self.logger.warning(f"Ollama request failed with status {response.status_code}")
                    return self._format_table(table)

                except requests.exceptions.RequestException as e:
                    if attempt < max_retries - 1:
                        time.sleep(1 * (attempt + 1))
                        continue
                    self.logger.warning(f"All Ollama request attempts failed: {e}")
                    return self._format_table(table)

            self.logger.warning(f"Ollama request failed after {max_retries} attempts")
            return self._format_table(table)

        except Exception as e:
            self.logger.warning(f"Error analyzing table: {e}")
            return self._format_table(table)

    def _find_table_context(self, table: Dict, contents: List[str]) -> str:
        """
        Find relevant context for a table from surrounding content.
        Enhanced with better context matching.

        Args:
            table (Dict): The table to find context for
            contents (List[str]): List of content strings to search through

        Returns:
            str: Relevant context for the table
        """
        if not contents or not table.get('headers'):
            return ""

        # Extract key terms from table headers and first row if available
        key_terms = set()
        for header in table['headers']:
            key_terms.update(word.lower() for word in header.split())

        if table.get('rows') and table['rows'][0]:
            first_row = table['rows'][0]
            for cell in first_row:
                if isinstance(cell, str):
                    key_terms.update(word.lower() for word in cell.split())

        # Find paragraphs with matching terms
        related_paragraphs = []
        for content in contents:
            content_words = set(word.lower() for word in content.split())
            matching_terms = len(key_terms & content_words)
            if matching_terms >= 2:  # Require at least 2 matching terms
                related_paragraphs.append((matching_terms, content))

        # Sort by relevance (number of matching terms) and take top 2
        related_paragraphs.sort(reverse=True)
        context = " ".join(content for _, content in related_paragraphs[:2])

        if len(context) > 500:
            context = context[:497] + "..."

        return context

    def _process_html(self, html_content: str, file_name: str) -> List[Document]:
        """
        Process HTML content into documents with enhanced table analysis.
        Updated to handle different content types more effectively.

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
            max_chunk_size = 500

            for content in contents:
                content_size = len(content)
                if current_chunk_size + content_size > max_chunk_size and current_chunk:
                    # Create document from current chunk
                    chunk_text = '\n'.join(current_chunk)
                    print(f"Chunk size: {len(chunk_text)} - {file_name}")
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

        # Process tables with enhanced context
        tables = parser.get_tables()
        for idx, table in enumerate(tables):
            try:
                # Find most relevant context for this table
                context = self._find_table_context(table, contents)

                # Attempt table analysis with retries
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
                        time.sleep(1 * (attempt + 1))

                if not table_analysis:
                    table_analysis = self._format_table(table)

                table_content = f"""
                Table Analysis:
                {table_analysis}

                Raw Table Data:
                {self._format_table(table)}

                Related Context:
                {context}
                """

                documents.append(Document(
                    page_content=table_content.strip(),
                    metadata={
                        "source": file_name,
                        "type": "table",
                        "table_index": idx,
                        "has_analysis": bool(table_analysis)
                    }
                ))
            except Exception as e:
                self.logger.warning(f"Error processing table {idx} in {file_name}: {e}")
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
                context = self._get_image_context(image, contents)
                image_content = f"""
                Image Content:
                {image['alt']}

                Related Context:
                {context}
                """

                documents.append(Document(
                    page_content=image_content.strip(),
                    metadata={
                        "source": file_name,
                        "type": "image",
                        "image_index": idx
                    }
                ))

        return documents

    def _process_documents(self, documents: List[Document]) -> List[Document]:
        """
        Process documents into sentence-level chunks.
        Updated with improved error handling and chunk processing.

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
                # Special handling for different content types
                if doc.metadata.get('type') == 'table':
                    # Don't split tables into sentences
                    processed_docs.append(doc)
                    continue

                if doc.metadata.get('type') == 'image':
                    # Don't split image descriptions into sentences
                    processed_docs.append(doc)
                    continue

                # Process main content into sentences
                nlp_doc = nlp(doc.page_content)

                current_chunk = []
                current_chunk_size = 0
                max_chunk_size = 500

                for sentence in nlp_doc.sentences:
                    sentence_text = sentence.text.strip()
                    if not sentence_text:
                        continue

                    sentence_size = len(sentence_text)

                    if current_chunk_size + sentence_size > max_chunk_size and current_chunk:
                        # Create document from current chunk
                        chunk_text = ' '.join(current_chunk)
                        processed_docs.append(Document(
                            page_content=chunk_text,
                            metadata={
                                **doc.metadata,
                                "chunk_size": len(chunk_text)
                            }
                        ))
                        current_chunk = []
                        current_chunk_size = 0

                    current_chunk.append(sentence_text)
                    current_chunk_size += sentence_size

                # Handle remaining chunk
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    processed_docs.append(Document(
                        page_content=chunk_text,
                        metadata={
                            **doc.metadata,
                            "chunk_size": len(chunk_text)
                        }
                    ))

            except Exception as e:
                self.logger.error(f"Error processing document: {e}")
                if doc.page_content.strip():
                    processed_docs.append(doc)

        return processed_docs

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
        alt_words = set(image['alt'].lower().split())

        # Score each content block based on relevance
        scored_contents = []
        for content in contents:
            content_lower = content.lower()
            score = sum(1 for word in alt_words if word in content_lower)
            if score > 0:
                scored_contents.append((score, content))

        # Sort by relevance score and take the most relevant
        if scored_contents:
            scored_contents.sort(reverse=True)
            context = scored_contents[0][1]
            if len(context) > 300:  # Limit context length
                context = context[:297] + "..."
            return context

        return ""

    def get_or_create_vectorstore(self) -> Optional[DatabricksVectorSearch]:
        """
        Get existing vectorstore if valid, otherwise create new one.
        Updated with better error handling and status reporting.

        Returns:
            Optional[DatabricksVectorSearch]: The vectorstore instance or None if creation fails
        """
        try:
            if self._check_vectorstore_validity():
                st.info("Loading existing vector search index...")
                return self.vector_store

            st.warning("Need to process HTML files. This may take a few minutes...")
            vectorstore, _ = self.ingest_html()
            return vectorstore

        except Exception as e:
            self.logger.error(f"Error accessing vector store: {e}")
            st.error("Failed to access vector store. Will attempt reprocessing.")
            return None

    def _add_documents_batch(self, documents: List[Dict]) -> None:
        """
        Add a batch of documents to the vector store.
        Updated to handle Databricks Vector Search requirements.

        Args:
            documents (List[Dict]): List of document dictionaries to add
        """
        try:
            if not documents:
                return

            # Generate embeddings for the batch
            texts = [doc['text'] for doc in documents]
            embeddings = self.embeddings.embed_documents(texts)

            # Create records with the correct schema
            records = []
            for doc, embedding in zip(documents, embeddings):
                record = {
                    "id": str(uuid4()),
                    "text": doc['text'],
                    "text_vector": embedding,
                    "source": doc['source'],
                    "type": doc.get('type', 'content'),
                }
                records.append(record)

            # Convert to DataFrame
            df = pd.DataFrame(records)

            try:
                # Add to vector store using Databricks specific method
                self.vector_store.add_texts(
                    texts=texts,
                    metadata=[{
                        'id': record['id'],
                        'source': record['source'],
                        'type': record['type']
                    } for record in records]
                )
                self.logger.info(f"Successfully added batch of {len(records)} documents")
            except Exception as e:
                self.logger.error(f"Error adding batch to vector store: {e}")
                # Try alternative direct DataFrame approach if available
                try:
                    self.vector_store._add_records_direct(df)
                    self.logger.info(f"Successfully added batch using direct method")
                except Exception as direct_e:
                    self.logger.error(f"Error adding batch using direct method: {direct_e}")
                    raise

        except Exception as e:
            self.logger.error(f"Error in _add_documents_batch: {e}")
            raise

    def ingest_html(self) -> Optional[Tuple[DatabricksVectorSearch, int]]:
        """
        Ingest HTML files into Databricks Vector Search.
        Updated with improved error handling and batching.

        Returns:
            Optional[Tuple[DatabricksVectorSearch, int]]: Tuple of (vectorstore, document_count) or None if ingestion fails
        """
        try:
            # Load cache information
            cache_info = self._load_cache_info()
            processed_files = cache_info["processed_files"]
            total_processed = 0
            documents = []
            updated = False

            # Set up progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Process HTML files
            html_files = [f for f in os.listdir(self.html_directory) if f.endswith('.html')]

            for idx, file_name in enumerate(html_files):
                file_path = self.html_directory / file_name
                current_modified = os.path.getmtime(file_path)

                if file_name in processed_files and processed_files[file_name] == current_modified:
                    continue

                try:
                    status_text.text(f"Processing {file_name}...")

                    with open(file_path, 'r', encoding='utf-8') as file:
                        html_content = file.read()

                    # Process the HTML content
                    docs = self._process_html(html_content, file_name)
                    processed_docs = self._process_documents(docs)

                    # Convert documents to the required format
                    for doc in processed_docs:
                        text_content = doc.page_content.strip()
                        if text_content:
                            doc_data = {
                                "text": text_content,
                                "source": doc.metadata.get("source", ""),
                                "type": doc.metadata.get("type", "content"),
                            }
                            documents.append(doc_data)

                    total_processed += len(processed_docs)
                    processed_files[file_name] = current_modified
                    updated = True

                    # Batch processing
                    if len(documents) >= self.batch_size:
                        try:
                            self._add_documents_batch(documents)
                            documents = []

                            # Update cache after successful batch
                            vectorstore_hash = str(hash(frozenset(processed_files.items())))
                            self._save_cache_info(processed_files, vectorstore_hash)
                        except Exception as batch_e:
                            self.logger.error(f"Error processing batch: {batch_e}")
                            # Continue with next batch
                            documents = []

                    progress = (idx + 1) / len(html_files)
                    progress_bar.progress(progress)
                    status_text.text(f"Processed {total_processed} documents from {idx + 1}/{len(html_files)} files")

                except Exception as e:
                    self.logger.error(f"Error processing {file_name}: {e}")
                    st.error(f"Error processing {file_name}: {str(e)}")
                    continue

            # Process remaining documents
            if documents:
                try:
                    self._add_documents_batch(documents)
                    if updated:
                        vectorstore_hash = str(hash(frozenset(processed_files.items())))
                        self._save_cache_info(processed_files, vectorstore_hash)
                except Exception as e:
                    self.logger.error(f"Error processing final batch: {e}")

            progress_bar.empty()
            status_text.empty()

            return self.vector_store, total_processed

        except Exception as e:
            self.logger.error(f"Error during ingestion: {e}")
            st.error(f"Error during document ingestion: {str(e)}")
            return None, 0

    def add_documents(self, documents: List[Document]) -> Optional[Tuple[DatabricksVectorSearch, int]]:
        """
        Add new documents to the existing vectorstore.
        Updated with improved error handling.

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
                self._add_documents_batch(doc_dicts)

            return self.vector_store, len(doc_dicts)

        except Exception as e:
            self.logger.error(f"Error adding documents: {e}")
            return None, 0