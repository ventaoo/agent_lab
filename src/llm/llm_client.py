import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import openai
import requests
from typing import List, Dict, Any
import yaml

class LLMClient:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.provider = self.config['llm']['provider']
        
        if self.provider == "local":
            self.model_name = self.config['llm']['local_model_path']
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.model.eval()
        elif self.provider == "api":
            self.api_key = self.config['llm']['api_key']
            openai.api_key = self.api_key
            openai.base_url = "https://api.deepseek.com"
    
    def generate(self, prompt: str, system_prompt: str = None) -> str:
        if self.provider == "local":
            return self._generate_local(prompt, system_prompt)
        elif self.provider == "api":
            return self._generate_api(prompt, system_prompt)
    
    def _generate_local(self, prompt: str, system_prompt: str = None) -> str:
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        inputs = self.tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config['llm']['max_tokens'],
                temperature=self.config['llm']['temperature'],
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the input part from the response
        response = response[len(full_prompt):].strip()
        return response
    
    def _generate_api(self, prompt: str, system_prompt: str = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = openai.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=self.config['llm']['temperature'],
            max_tokens=self.config['llm']['max_tokens']
        )
        
        return response.choices[0].message.content.strip()

# Example usage
if __name__ == "__main__":
    client = LLMClient()
    response = client.generate("Hello, how are you?")
    print(response)