import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import cosine
from typing import List
import warnings
warnings.filterwarnings('ignore')

class DriftMetrics:
    @staticmethod
    def calculate_kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
        """Calculate KL divergence between two probability distributions"""
        p = np.asarray(p, dtype=float)
        q = np.asarray(q, dtype=float)
        
        # Add small epsilon to avoid log(0)
        p = np.clip(p, epsilon, 1.0)
        q = np.clip(q, epsilon, 1.0)
        
        # Normalize if not already
        p = p / p.sum() if p.sum() != 0 else p
        q = q / q.sum() if q.sum() != 0 else q
        
        return entropy(p, q)
    
    @staticmethod
    def calculate_embedding_shift(embeddings: List[np.ndarray]) -> float:
        """Calculate average cosine distance between consecutive embeddings"""
        if len(embeddings) < 2:
            return 0.0
        
        distances = []
        for i in range(1, len(embeddings)):
            # Ensure embeddings are 1D arrays
            emb1 = embeddings[i-1].flatten()
            emb2 = embeddings[i].flatten()
            
            # Normalize embeddings
            emb1 = emb1 / np.linalg.norm(emb1) if np.linalg.norm(emb1) != 0 else emb1
            emb2 = emb2 / np.linalg.norm(emb2) if np.linalg.norm(emb2) != 0 else emb2
            
            dist = cosine(emb1, emb2)
            distances.append(dist)
        
        return np.mean(distances)
    
    @staticmethod
    def calculate_response_similarity(response_embeddings: List[np.ndarray]) -> float:
        """Calculate average similarity between recent responses"""
        if len(response_embeddings) < 2:
            return 1.0  # No drift if only one response
        
        similarities = []
        for i in range(1, len(response_embeddings)):
            # Normalize embeddings
            emb1 = response_embeddings[i-1].flatten()
            emb2 = response_embeddings[i].flatten()
            
            emb1 = emb1 / np.linalg.norm(emb1) if np.linalg.norm(emb1) != 0 else emb1
            emb2 = emb2 / np.linalg.norm(emb2) if np.linalg.norm(emb2) != 0 else emb2
            
            sim = 1 - cosine(emb1, emb2)
            similarities.append(max(0, sim))  # Ensure non-negative
        
        return np.mean(similarities)