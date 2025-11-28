from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings
from llama_index.core.node_parser import SimpleNodeParser


Settings.llm = Ollama(model="qwen2.5-coder:3b", request_timeout=120.0)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")


CHUNK_SIZE = 256          # smaller chunks -> smaller embedding payloads
CHUNK_OVERLAP = 40
EMBED_BATCH_SIZE = 4      # keep small to avoid OOM / server crashes
MAX_FILE_CHARS = 150_000  # skip files larger than this (tune if needed)

IOS_PROJECT_DIR = "ios_project"
PERSIST_DIR = "./data"

def _collect_documents(path):
    # Use SimpleDirectoryReader to read allowed extensions, then filter large docs
    reader = SimpleDirectoryReader(
        path,
        recursive=True,
        required_exts=[".swift", ".md", ".txt", ".json"]
    )
    docs = reader.load_data()

    # Filter out very large files which cause embedding crashes
    filtered = []
    for d in docs:
        # LlamaIndex Document may expose get_text() or .text
        text = getattr(d, "get_text", None)
        if callable(text):
            txt = d.get_text()
        else:
            txt = getattr(d, "text", None) or ""

        if len(txt) > MAX_FILE_CHARS:
            print(f"Skipping very large file chunked as one doc (len={len(txt)}). Consider breaking this file: {getattr(d, 'doc_id', 'unknown')}")
            continue
        filtered.append(d)
    return filtered


def create_index():
    print("Collecting documents...")
    docs = _collect_documents(IOS_PROJECT_DIR)

    # Create nodes with small chunk sizes
    parser = SimpleNodeParser.from_defaults(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    print("Splitting into nodes...")
    nodes = parser.get_nodes_from_documents(docs)

    # Use local embedding model and LLM (Settings already set above)
    embed_model = OllamaEmbedding(model_name="nomic-embed-text")
    llm = Ollama(model="qwen2.5-coder:3b", request_timeout=120.0)

    print("Building index (this will take a while)...")
    index = VectorStoreIndex.from_documents(
        nodes,
        llm=llm,
        embed_model=embed_model,
        embed_batch_size=EMBED_BATCH_SIZE
    )

    # Persist index safely
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    print("Index created and persisted to", PERSIST_DIR)


if __name__ == "__main__":
    create_index()
