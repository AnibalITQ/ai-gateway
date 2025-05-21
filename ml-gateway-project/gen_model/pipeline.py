from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch


model_name = "tiiuae/falcon-rw-1b"

tokenizer = AutoTokenizer.from_pretrained(model_name)
generator = pipeline(
    "text-generation",
    model=model_name,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

def run_generation(petition):
    try:
        sequences = generator(
            petition.prompt,
            max_length=200,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
        )
        output = sequences[0]["generated_text"]
        return {"output": output}
    except Exception as e:
        return {"error": str(e)}