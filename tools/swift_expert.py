from llama_index.core import StorageContext, load_index_from_storage

class SwiftExpert:
    def __init__(self, persist_dir="./data"):
        storage = StorageContext.from_defaults(persist_dir=persist_dir)
        self.index = load_index_from_storage(storage)

    def answer(self, question: str) -> str:
        engine = self.index.as_query_engine(similarity_top_k=4)
        response = engine.query(question)
        return str(response)
    
