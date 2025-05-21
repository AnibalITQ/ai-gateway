from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

class STTModel:
    def __init__(self):
        self.model_name = "openai/whisper-tiny"
        self.processor = WhisperProcessor.from_pretrained(self.model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name)
        
    def transcribe(self, audio_array: torch.Tensor) -> dict:
        inputs = self.processor(
            audio_array, 
            sampling_rate=16000, 
            return_tensors="pt"
        )
        
        with torch.no_grad():
            generated_ids = self.model.generate(inputs.input_features)
            
        transcription = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )[0]
        
        return {"transcription": transcription}