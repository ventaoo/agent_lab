import numpy as np
from typing import List, Dict
from ..metrics.drift_metrics import DriftMetrics

class PromptUpdater:
    def __init__(self, drift_threshold: float = 0.1):
        self.drift_threshold = drift_threshold
        self.drift_metrics = DriftMetrics()
        self.prompt_history = []
    
    def detect_drift(self, agent_responses: List[str], embedding_model) -> Dict[str, float]:
        """Detect drift in agent responses"""
        if len(agent_responses) < 2:
            return {"kl_divergence": 0.0, "embedding_shift": 0.0, "similarity": 1.0}
        
        # Calculate embeddings for responses
        embeddings = [embedding_model.encode(resp) for resp in agent_responses[-50:]]  # Use last 50 responses
        
        # Calculate KL divergence (using normalized response lengths as proxy for distribution)
        response_lengths = [len(resp) for resp in agent_responses[-50:]]
        if len(set(response_lengths)) > 1:  # Only calculate if we have different lengths
            recent_lengths = response_lengths[-10:] if len(response_lengths) >= 10 else response_lengths
            historical_lengths = response_lengths[:-10] if len(response_lengths) >= 10 else [np.mean(response_lengths)]
            
            # Normalize to create probability distributions
            recent_dist = np.array(recent_lengths) / sum(recent_lengths)
            historical_dist = np.array(historical_lengths) / sum(historical_lengths)
            
            # Pad if necessary
            if len(recent_dist) != len(historical_dist):
                max_len = max(len(recent_dist), len(historical_dist))
                recent_dist = np.pad(recent_dist, (0, max_len - len(recent_dist)), mode='constant')
                historical_dist = np.pad(historical_dist, (0, max_len - len(historical_dist)), mode='constant')
            
            kl_div = self.drift_metrics.calculate_kl_divergence(recent_dist, historical_dist)
        else:
            kl_div = 0.0
        
        # Calculate embedding shift
        embedding_shift = self.drift_metrics.calculate_embedding_shift(embeddings)
        
        # Calculate response similarity
        similarity = self.drift_metrics.calculate_response_similarity(embeddings)
        
        return {
            "kl_divergence": kl_div,
            "embedding_shift": embedding_shift,
            "similarity": similarity
        }
    
    def update_prompt(self, agent_id: int, current_prompt: str, drift_metrics: Dict[str, float]) -> str:
        """Update prompt based on drift metrics"""
        drift_score = max(drift_metrics.values())
        
        if drift_score > self.drift_threshold:
            # Add more structured guidance to reduce drift
            updated_prompt = f"""
            {current_prompt}
            
            IMPORTANT: Maintain consistent response style and format.
            - Keep responses concise but informative
            - Follow the same reasoning pattern as before
            - Avoid drifting from established response patterns
            """
            return updated_prompt
        else:
            return current_prompt