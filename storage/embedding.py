from typing import List
from llama_cpp import LLama

EMBEDDING_MODEL_PATH = 'D:/models/BGE/bge-large-en-v1.5-q8_0.gguf'

embedding_llm = LLama(model_path=EMBEDDING_MODEL_PATH, embedding=True)


def generate_embedding(texts: List[str]) -> List[float]:

    return embedding_llm.create_embedding(texts)
