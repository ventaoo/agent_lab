import faiss
import numpy as np
import yaml
from sentence_transformers import SentenceTransformer
import json
import os
from typing import List, Dict, Any
import pickle

class VectorMemory:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)  # Changed from json.load to yaml.safe_load
        
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = self.config['metrics']['embedding_dim']
        self.max_memory_size = self.config['metrics']['max_memory_size']
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        
        # Store metadata
        self.memory_store = []  # List of dictionaries: {'text': str, 'embedding': np.array, 'metadata': dict}
        self.persist_to_disk = self.config['memory']['persist_to_disk']
        self.memory_path = self.config['memory']['memory_path']
        
        if self.persist_to_disk and os.path.exists(self.memory_path):
            self.load_memory()
    
    def add_entry(self, text: str, metadata: Dict[str, Any] = None):
        embedding = self.embedding_model.encode([text])[0]
        embedding = embedding / np.linalg.norm(embedding)  # Normalize for cosine similarity
        
        # Add to memory store
        entry = {
            'text': text,
            'embedding': embedding.astype('float32'),
            'metadata': metadata or {}
        }
        self.memory_store.append(entry)
        
        # Add to FAISS index
        self.index.add(embedding.reshape(1, -1))
        
        # Maintain size limit
        if len(self.memory_store) > self.max_memory_size:
            removed_entry = self.memory_store.pop(0)
            # Note: FAISS doesn't support deletion, so we just keep track of valid entries
            # In a real implementation, we'd need to rebuild the index periodically
        
        if self.persist_to_disk:
            self.save_memory()
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        query_embedding = self.embedding_model.encode([query])[0]
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Search in FAISS
        scores, indices = self.index.search(query_embedding.reshape(1, -1), k)
        
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx < len(self.memory_store):
                result = {
                    'text': self.memory_store[idx]['text'],
                    'score': float(scores[0][i]),
                    'metadata': self.memory_store[idx]['metadata']
                }
                results.append(result)
        
        return results
    
    def get_average_embedding(self) -> np.ndarray:
        if not self.memory_store:
            return np.zeros(self.dimension)
        
        embeddings = np.array([entry['embedding'] for entry in self.memory_store])
        return np.mean(embeddings, axis=0)
    
    def save_memory(self):
        os.makedirs(self.memory_path, exist_ok=True)
        # Save embeddings and texts separately for efficiency
        embeddings = np.array([entry['embedding'] for entry in self.memory_store])
        texts = [entry['text'] for entry in self.memory_store]
        metadata = [entry['metadata'] for entry in self.memory_store]
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(self.memory_path, "faiss_index.bin"))
        
        # Save other data
        with open(os.path.join(self.memory_path, "memory_data.pkl"), 'wb') as f:
            pickle.dump({
                'embeddings': embeddings,
                'texts': texts,
                'metadata': metadata
            }, f)
    
    def load_memory(self):
        try:
            # Load FAISS index
            self.index = faiss.read_index(os.path.join(self.memory_path, "faiss_index.bin"))
            
            # Load other data
            with open(os.path.join(self.memory_path, "memory_data.pkl"), 'rb') as f:
                data = pickle.load(f)
            
            # Reconstruct memory store
            for i, text in enumerate(data['texts']):
                entry = {
                    'text': text,
                    'embedding': data['embeddings'][i],
                    'metadata': data['metadata'][i]
                }
                self.memory_store.append(entry)
        except Exception as e:
            print(f"Error loading memory: {e}")
            # Initialize empty if loading fails
            self.index = faiss.IndexFlatIP(self.dimension)
            self.memory_store = []