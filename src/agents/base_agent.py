import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
import numpy as np
from sentence_transformers import SentenceTransformer
from .memory import AgentMemory
import os
from ..utils.config import Config

class BaseAgent:
    def __init__(self, agent_id: int, config: Config):
        self.agent_id = agent_id
        self.config = config
        self.use_local = config.get('model.use_local', True)
        
        # Initialize model
        if self.use_local:
            self.model_name = config.get('model.local_model_name')
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            if torch.cuda.is_available():
                self.model = self.model.cuda()
        else:
            self.client = OpenAI(
                api_key=config.get('model.api_key'),
                base_url=config.get('model.api_base')
            )
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize memory
        memory_size = config.get('experiment.memory_size', 100)
        self.memory = AgentMemory(memory_size=memory_size)
        
        # Track prompt evolution
        self.prompt_history = []
        self.response_history = []
    
    def generate_response(self, query: str, context: str = "") -> str:
        if self.use_local:
            return self._generate_local(query, context)
        else:
            return self._generate_api(query, context)
    
    def _generate_local(self, query: str, context: str) -> str:
        prompt = f"Context: {context}\nQuery: {query}\nResponse:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.7,
                do_sample=True
            )
        
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response.strip()
    
    def _generate_api(self, query: str, context: str) -> str:
        prompt = f"Context: {context}\nQuery: {query}\nResponse:"
        
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=128
        )
        
        return response.choices[0].message.content.strip()
    
    def update_memory(self, query: str, response: str):
        # Generate embedding for the interaction
        embedding_text = f"Query: {query} Response: {response}"
        embedding = self.embedding_model.encode(embedding_text)
        
        self.memory.add_interaction(query, response, embedding)
        self.response_history.append(response)
    
    def get_embedding(self, text: str) -> np.ndarray:
        return self.embedding_model.encode(text)