from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "Bronsn/ganda_llama_8b_64"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map={"": "mps"}
)

prompt = "Translate the following text to Luganda: Good morning! How are you doing? My name is James and I am French"
inputs = tokenizer(prompt, return_tensors="pt").to("mps")

output = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(output[0], skip_special_tokens=True))

