import numpy as np
import yaml
from scipy.stats import entropy
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import json

class DriftMetrics:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)  # Changed from json.load to yaml.safe_load
        
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.baseline_prompts = {}  # Store baseline embeddings for each worker
        self.worker_history = {}   # Store recent responses for each worker
    
    def calculate_success_rate(self, responses: List[Dict[str, Any]]) -> float:
        """Calculate success rate based on response quality"""
        if not responses:
            return 0.0
        
        success_count = 0
        for response in responses:
            # Simple heuristic: check if response is not empty and contains meaningful content
            if response.get('response') and len(response['response'].strip()) > 10:
                success_count += 1
        
        return success_count / len(responses)
    
    def calculate_kl_divergence(self, current_responses: List[str], baseline_responses: List[str]) -> float:
        """Calculate KL divergence between current and baseline response distributions"""
        if not current_responses or not baseline_responses:
            return 0.0
        
        # Combine all responses for vocabulary
        all_responses = current_responses + baseline_responses
        vectorizer = CountVectorizer(max_features=1000, stop_words='english')
        doc_term_matrix = vectorizer.fit_transform(all_responses).toarray()
        
        # Get word counts for current and baseline
        current_matrix = doc_term_matrix[:len(current_responses)]
        baseline_matrix = doc_term_matrix[len(current_responses):]
        
        # Calculate average word distributions
        current_dist = np.mean(current_matrix, axis=0)
        baseline_dist = np.mean(baseline_matrix, axis=0)
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        current_dist = current_dist + epsilon
        baseline_dist = baseline_dist + epsilon
        
        # Normalize
        current_dist = current_dist / np.sum(current_dist)
        baseline_dist = baseline_dist / np.sum(baseline_dist)
        
        # Calculate KL divergence
        kl_div = entropy(current_dist, baseline_dist)
        return float(kl_div)
    
    def calculate_embedding_shift(self, current_response: str, worker_id: str) -> float:
        """Calculate embedding shift from historical average"""
        if worker_id not in self.worker_history:
            self.worker_history[worker_id] = []
        
        current_embedding = self.embedding_model.encode([current_response])[0]
        current_embedding = current_embedding / np.linalg.norm(current_embedding)
        
        if not self.worker_history[worker_id]:
            # First response, no shift possible
            self.worker_history[worker_id].append(current_embedding)
            return 0.0
        
        # Calculate average historical embedding
        hist_embeddings = np.array(self.worker_history[worker_id])
        avg_hist_embedding = np.mean(hist_embeddings, axis=0)
        avg_hist_embedding = avg_hist_embedding / np.linalg.norm(avg_hist_embedding)
        
        # Calculate cosine similarity (1 - cosine distance)
        cosine_sim = np.dot(current_embedding, avg_hist_embedding)
        shift = 1.0 - cosine_sim  # Higher value means more shift
        
        # Add current embedding to history
        self.worker_history[worker_id].append(current_embedding)
        
        # Keep only recent history
        if len(self.worker_history[worker_id]) > 50:
            self.worker_history[worker_id] = self.worker_history[worker_id][-50:]
        
        return float(shift)
    
    def detect_drift(self, worker_id: str, responses: List[Dict[str, Any]], 
                    baseline_responses: List[str], drift_threshold: float = 0.15) -> bool:
        """Detect if drift has occurred based on multiple metrics"""
        if not responses:
            return False
        
        # Calculate metrics
        success_rate = self.calculate_success_rate(responses)
        kl_div = self.calculate_kl_divergence([r['response'] for r in responses], baseline_responses)
        
        # Calculate embedding shift for the most recent response
        if responses:
            embedding_shift = self.calculate_embedding_shift(responses[-1]['response'], worker_id)
        else:
            embedding_shift = 0.0
        
        # Drift detection logic
        drift_detected = (
            kl_div > drift_threshold or 
            embedding_shift > drift_threshold or
            success_rate < 0.5  # Success rate too low
        )
        
        return drift_detected
    
    def get_metrics_summary(self, worker_id: str, responses: List[Dict[str, Any]], 
                           baseline_responses: List[str]) -> Dict[str, float]:
        """Get summary of all metrics for reporting"""
        if not responses:
            return {
                'success_rate': 0.0,
                'kl_divergence': 0.0,
                'embedding_shift': 0.0
            }
        
        success_rate = self.calculate_success_rate(responses)
        kl_div = self.calculate_kl_divergence([r['response'] for r in responses], baseline_responses)
        
        # Calculate embedding shift for the most recent response
        embedding_shift = self.calculate_embedding_shift(responses[-1]['response'], worker_id)
        
        return {
            'success_rate': success_rate,
            'kl_divergence': kl_div,
            'embedding_shift': embedding_shift
        }