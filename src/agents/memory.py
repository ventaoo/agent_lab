import numpy as np
from typing import List, Dict, Any
import faiss

class AgentMemory:
    def __init__(self, memory_size: int = 100, embedding_dim: int = 384):
        self.memory_size = memory_size
        self.embedding_dim = embedding_dim
        self.memory: List[Dict[str, Any]] = []
        self.index = faiss.IndexFlatIP(embedding_dim)
        
    def add_interaction(self, query: str, response: str, embedding: np.ndarray):
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)
        
        self.memory.append({
            'query': query,
            'response': response,
            'timestamp': len(self.memory),
            'embedding': embedding
        })
        
        # Update FAISS index
        embedding = embedding / np.linalg.norm(embedding)  # normalize
        self.index.add(embedding.reshape(1, -1))
    
    def get_recent_interactions(self, n: int = 5) -> List[Dict[str, Any]]:
        return self.memory[-n:] if len(self.memory) >= n else self.memory
    
    def find_similar(self, query_embedding: np.ndarray, k: int = 3) -> List[Dict[str, Any]]:
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        scores, indices = self.index.search(query_embedding.reshape(1, -1), k)
        
        results = []
        for idx in indices[0]:
            if idx < len(self.memory):
                results.append(self.memory[idx])
        return results