from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch

class QAModel:
    def __init__(self):
        self.model_name = "distilbert-base-uncased-distilled-squad"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)
        
    def predict(self, question: str, context: str) -> dict:
        inputs = self.tokenizer(
            question,
            context,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            stride=128
        )
        
        outputs = self.model(**inputs)
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits)
        
        answer = self.tokenizer.convert_tokens_to_string(
            self.tokenizer.convert_ids_to_tokens(
                inputs["input_ids"][0][answer_start:answer_end+1]
            )
        )
        
        return {"answer": answer}