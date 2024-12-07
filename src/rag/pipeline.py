# src/rag/pipeline.py
import json
from pathlib import Path
from typing import List
import logging

class RAGPipeline:
    def __init__(self, config, embed_model, vector_store, qwen_model, logger=None):
        self.config = config
        self.embed_model = embed_model
        self.vector_store = vector_store
        self.qwen_model = qwen_model
        self.logger = logger or logging.getLogger(__name__)

        with open(self.config["paths"]["metadata_index"], "r", encoding='utf-8') as f:
            self.metadata = json.load(f)

        with open(self.config["paths"]["processed_docs"], "r", encoding='utf-8') as f:
            self.all_docs = json.load(f)

        # Define max context tokens based on model's capability
        self.max_context_tokens = self.config["model"].get("max_context_tokens", 3000)  # Adjust based on model's max tokens
        self.chunk_size = self.config["model"].get("chunk_size", 1000)                    # Size per chunk
        self.overlap = self.config["model"].get("overlap_tokens", 200)                    # Overlap between chunks

    def retrieve_context(self, query: str):
        top_n = self.config["rag_integration"]["retrieval_top_n"]
        q_emb = self.embed_model.encode([query])
        D, I = self.vector_store.search(q_emb, top_n)

        retrieved_docs = []
        for idx in I[0]:
            if idx < 0:
                continue
            meta = self.metadata[idx]
            doc_path = meta["path"]
            for d in self.all_docs:
                if d["path"] == doc_path:
                    retrieved_docs.append(d["text"])
                    break

        # Combine all retrieved texts into a single context
        combined_context = "\n\n".join(retrieved_docs)
        return combined_context

    def tokenize_length(self, text: str) -> int:
        # Accurate token count using the model's tokenizer
        inputs = self.qwen_model.tokenizer(text, return_tensors='pt')
        return inputs.input_ids.shape[1]

    def split_into_chunks_by_tokens(self, text: str, max_tokens: int, overlap_tokens: int) -> List[str]:
        """
        Split text into chunks based on tokens.
        """
        input_ids = self.qwen_model.tokenizer(text, return_tensors='pt').input_ids[0]
        chunks = []
        start = 0
        length = input_ids.shape[0]

        while start < length:
            end = start + max_tokens
            chunk_ids = input_ids[start:end]
            chunk_text = self.qwen_model.tokenizer.decode(chunk_ids, skip_special_tokens=True)
            chunks.append(chunk_text)
            start = end - overlap_tokens
            if start < 0:
                start = 0
        return chunks

    def answer_query(self, query: str):
        self.logger.info(f"Processing query: {query}")
        context = self.retrieve_context(query)

        # Check token length
        total_tokens = self.tokenize_length(context)
        if total_tokens > self.max_context_tokens:
            self.logger.info("Context too long, splitting into chunks.")
            # Split into chunks
            max_tokens_for_chunk = self.chunk_size
            overlap_tokens = self.overlap
            chunks = self.split_into_chunks_by_tokens(context, max_tokens_for_chunk, overlap_tokens)

            # Concatenate chunks up to the model's token limit
            combined_chunks = []
            current_length = 0
            for chunk in chunks:
                chunk_length = self.tokenize_length(chunk)
                if current_length + chunk_length <= self.max_context_tokens:
                    combined_chunks.append(chunk)
                    current_length += chunk_length
                else:
                    break
            combined_context = "\n\n".join(combined_chunks)

            # Generate a single answer based on the combined context
            self.logger.info("Generating answer based on combined context.")
            final_answer = self.qwen_model.generate_answer(query, combined_context)
        else:
            self.logger.info("Context fits within the token limit, generating answer.")
            # Single answer
            final_answer = self.qwen_model.generate_answer(query, context)

        return final_answer
