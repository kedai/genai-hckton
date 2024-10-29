#!/usr/bin/env python3
import os
import json
import logging
import requests
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import streamlit as st
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from dp_html_parser import HTMLContentParser

from config import nlp

class DocumentProcessor:
    def __init__(self, index_directory: str, html_directory: str, embeddings, llm):
        self.index_directory = Path(index_directory)
        self.html_directory = Path(html_directory)
        self.llm = llm.model
        self.embeddings = embeddings
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.batch_size = 50

        # Change cache file location to be alongside the index
        self.cache_directory = self.index_directory / "cache"
        self.cache_directory.mkdir(parents=True, exist_ok=True)
        self.processed_cache_file = self.cache_directory / "processed_files.json"
        self.vectorstore_info_file = self.cache_directory / "vectorstore_info.json"
        self.ollama_url = "http://localhost:11434/api/generate"

    def _load_cache_info(self) -> Dict:
        """Load both processed files cache and vectorstore info."""
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

    def _save_cache_info(self, processed_files: Dict, vectorstore_hash: str = None):
        """Save both processed files cache and vectorstore info."""
        try:
            with open(self.processed_cache_file, 'w') as f:
                json.dump(processed_files, f)

            vectorstore_info = {
                "vectorstore_hash": vectorstore_hash,
                "last_modified": datetime.now().isoformat()
            }
            with open(self.vectorstore_info_file, 'w') as f:
                json.dump(vectorstore_info, f)

        except Exception as e:
            self.logger.error(f"Error saving cache info: {e}")

    def _check_vectorstore_validity(self) -> bool:
        """Check if the existing vectorstore is valid and up-to-date."""
        if not os.path.exists(self.index_directory / "index.faiss"):
            return False

        try:
            cache_info = self._load_cache_info()

            if not cache_info.get("vectorstore_hash"):
                return False

            current_files = {f for f in os.listdir(self.html_directory) if f.endswith('.html')}
            cached_files = set(cache_info["processed_files"].keys())

            if current_files - cached_files:
                return False

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
        """Format table data in a more readable and structured way."""
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
        """Analyze table content using Ollama to extract meaning and relationships."""
        print(f"Analyzing table: {table}")
        try:
            headers = table['headers']
            rows = table['rows']

            if any('status' in h.lower() for h in headers):
                prompt = f"""
                Analyze this API status/state table:
                Headers: {', '.join(headers)}
                Data: {str(rows)}
                Context: {context}

                Please provide:
                1. What state transitions or status codes are described
                2. Key relationships between columns
                3. Important conditions or requirements
                """
            elif any('parameter' in h.lower() or 'field' in h.lower() for h in headers):
                prompt = f"""
                Analyze this API parameter/field table:
                Headers: {', '.join(headers)}
                Data: {str(rows)}
                Context: {context}

                Please provide:
                1. Key parameters/fields and their purposes
                2. Required vs optional fields
                3. Data type requirements or constraints
                """
            else:
                prompt = f"""
                Analyze this API documentation table:
                Headers: {', '.join(headers)}
                Data: {str(rows)}
                Context: {context}

                Please provide:
                1. Main purpose of this table
                2. Key relationships between columns
                3. Important patterns or requirements
                """

            response = requests.post(
                self.ollama_url,
                headers={"Content-Type": "application/json"},
                json={
                    "model": self.llm,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )

            if response.status_code == 200:
                try:
                    full_response = ""
                    for line in response.text.strip().split('\n'):
                        try:
                            data = json.loads(line)
                            full_response += data.get('response', '')
                        except json.JSONDecodeError:
                            continue
                    return full_response.strip()
                except Exception as e:
                    self.logger.warning(f"Error parsing Ollama response: {e}")
                    return self._format_table(table)
            else:
                self.logger.warning(f"Ollama request failed with status {response.status_code}")
                return self._format_table(table)

        except requests.exceptions.RequestException as e:
            self.logger.warning(f"Error connecting to Ollama: {e}")
            return self._format_table(table)
        except Exception as e:
            self.logger.warning(f"Unexpected error in table analysis: {e}")
            return self._format_table(table)

    def _process_html(self, html_content: str, file_name: str) -> List[Document]:
        """Process HTML content into documents with enhanced table analysis."""
        documents = []
        parser = HTMLContentParser(html_content)

        contents = parser.get_contents()
        content_text = '\n'.join(contents) if contents else ""

        if content_text:
            documents.append(Document(
                page_content=content_text,
                metadata={
                    "source": file_name,
                    "type": "main_content",
                    "chunk_size": len(content_text)
                }
            ))

        tables = parser.get_tables()
        for idx, table in enumerate(tables):
            try:
                context = self._find_table_context(table, contents)
                table_analysis = self._analyze_table_with_ollama(table, context)

                table_content = f"""
                Table Analysis:
                {table_analysis}

                Raw Table Data:
                {self._format_table(table)}
                """

                documents.append(Document(
                    page_content=table_content.strip(),
                    metadata={
                        "source": file_name,
                        "type": "table",
                        "table_index": idx,
                        "has_analysis": True
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

        images = parser.get_images()
        for idx, image in enumerate(images):
            if image['alt'].strip():
                context = self._get_image_context(image, contents)
                documents.append(Document(
                    page_content=f"Image description: {image['alt']}\nContext: {context}",
                    metadata={
                        "source": file_name,
                        "type": "image",
                        "image_index": idx
                    }
                ))

        return documents

    def _get_image_context(self, image: Dict, contents: List[str]) -> str:
        """Get surrounding text context for an image if available."""
        context = ""
        if image.get('alt') and contents:
            alt_words = set(image['alt'].lower().split())
            for content in contents:
                if any(word in content.lower() for word in alt_words):
                    context = content
                    break
        return context

    def _find_table_context(self, table: Dict, contents: List[str]) -> str:
        """Find relevant context for a table from surrounding content."""
        context = ""
        if not contents:
            return context

        table_words = set()
        for header in table['headers']:
            table_words.update(header.lower().split())

        related_paragraphs = []
        for content in contents:
            content_words = set(content.lower().split())
            if len(table_words & content_words) >= 2:
                related_paragraphs.append(content)

        if related_paragraphs:
            context = " ".join(related_paragraphs)
            if len(context) > 500:
                context = context[:497] + "..."

        return context

    def _process_documents(self, documents: List[Document]) -> List[Document]:
        """Process documents into sentence-level chunks."""
        processed_docs = []
        for doc in documents:
            if not doc.page_content.strip():
                continue

            try:
                nlp_doc = nlp(doc.page_content)
                for sentence in nlp_doc.sentences:
                    if sentence.text.strip():
                        processed_docs.append(Document(
                            page_content=sentence.text.strip(),
                            metadata=doc.metadata
                        ))
            except Exception as e:
                self.logger.error(f"Error processing document: {e}")
                continue

        return processed_docs

    def get_or_create_vectorstore(self) -> FAISS:
        """Get existing vectorstore if valid, otherwise create new one."""
        if self._check_vectorstore_validity():
            try:
                st.info("Loading existing vectorstore...")
                return FAISS.load_local(str(self.index_directory), self.embeddings, allow_dangerous_deserialization=True)
            except Exception as e:
                self.logger.error(f"Error loading existing vectorstore: {e}")

        st.warning("Need to process HTML files. This may take a few minutes...")
        vectorstore, _ = self.ingest_html()
        return vectorstore

    def ingest_html(self) -> Optional[Tuple[FAISS, int]]:
        """Ingest HTML files with persistent caching."""
        try:
            vectorstore = FAISS.from_documents([Document(page_content="", metadata={})], self.embeddings)
            cache_info = self._load_cache_info()
            processed_files = cache_info["processed_files"]
            total_processed = 0
            documents = []
            updated = False

            progress_bar = st.progress(0)
            status_text = st.empty()

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

                    docs = self._process_html(html_content, file_name)
                    processed_docs = self._process_documents(docs)
                    documents.extend(processed_docs)
                    total_processed += len(processed_docs)

                    processed_files[file_name] = current_modified
                    updated = True

                    if len(documents) >= self.batch_size:
                        self._update_vectorstore(vectorstore, documents)
                        documents = []
                        vectorstore_hash = str(hash(frozenset(processed_files.items())))
                        self._save_cache_info(processed_files, vectorstore_hash)

                    progress = (idx + 1) / len(html_files)
                    progress_bar.progress(progress)
                    status_text.text(f"Processed {total_processed} documents from {idx + 1}/{len(html_files)} files")

                except Exception as e:
                    self.logger.error(f"Error processing {file_name}: {e}")
                    st.error(f"Error processing {file_name}: {str(e)}")
                    continue

            if documents:
                self._update_vectorstore(vectorstore, documents)
                if updated:
                    vectorstore_hash = str(hash(frozenset(processed_files.items())))
                    self._save_cache_info(processed_files, vectorstore_hash)

            progress_bar.empty()
            status_text.empty()

            vectorstore.save_local(str(self.index_directory))
            return vectorstore, total_processed

        except Exception as e:
            self.logger.error(f"Error during ingestion: {e}")
            return None, 0

    def _update_vectorstore(self, vectorstore: FAISS, documents: List[Document]):
        """Update the vectorstore with new documents."""
        if documents:
            batch_vectorstore = FAISS.from_documents(documents, self.embeddings)
            vectorstore.merge_from(batch_vectorstore)
            vectorstore.save_local(str(self.index_directory))