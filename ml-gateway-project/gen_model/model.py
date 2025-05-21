from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class TextGenerator:
    def __init__(self):
        self.model_name = "tiiuae/falcon-rw-1b"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
    def generate(self, prompt: str) -> dict:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_length=200,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.95,
            do_sample=True
        )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"generated_text": generated_text}