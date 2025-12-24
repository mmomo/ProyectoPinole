from llama_index.core import StorageContext, load_index_from_storage
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings

import time
from ollama._types import ResponseError
import httpx

from prompts import REVIEW_PROMPT

Settings.llm = Ollama(model="qwen2.5-coder:3b", request_timeout=180.0)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

EMBED_MODEL_NAME = "nomic-embed-text"
LLM_MODEL_NAME = "qwen2.5-coder:3b"
PERSIST_DIR = "./data"

def ask(query, retries=1):
    storage = StorageContext.from_defaults(persist_dir=PERSIST_DIR)

    embed_model = OllamaEmbedding(model_name=EMBED_MODEL_NAME)
    llm = Ollama(model=LLM_MODEL_NAME, request_timeout=180.0)

    index = load_index_from_storage(storage, llm=llm, embed_model=embed_model)

    engine = index.as_query_engine(
        response_mode="compact",     # compact reduces LLM calls and size of prompt
        similarity_top_k=3          # smaller number of retrieved chunks -> faster
    )

    attempt = 0
    while True:
        try:
            response = engine.query(f"""
                Review Prompt:
                {REVIEW_PROMPT}
                
                Instruction:
                {query}
            """)
            print(response)
            break
        except (httpx.ReadTimeout, ResponseError) as e:
            attempt += 1
            print(f"Request failed with {type(e).__name__}: {e}")
            if attempt > retries:
                print("Exceeded retries — possible causes: Ollama crashed, model OOM, or long prompt. Try smaller query or switch embedding model.")
                raise
            backoff = 2 ** attempt
            print(f"Retrying in {backoff}s...")
            time.sleep(backoff)

if __name__ == "__main__":
    while True:
        q = input("\n> Hi paps, ask me anything: ")
        ask(q)
