from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# model_name = "Bronsn/ganda_llama_8b_64"
model_name = "Helsinki-NLP/opus-mt-en-lg"

tokenizer = AutoTokenizer.from_pretrained(model_name)
# "mps" is used for this mac (would have been "cuda" if there was a GPU)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map={"": "mps"}
)

prompt = "Translate the following text to Luganda: Good morning! How are you doing? My name is James and I am French"
# "mps" is used for this mac (would have been "cuda" if there was a GPU)
inputs = tokenizer(prompt, return_tensors="pt").to("mps")
print("input: ", inputs)
output = model.generate(**inputs, max_new_tokens=100)
print("output: ", output)
output2 = model(**inputs)
print("output2: ", output2)
print(tokenizer.decode(output[0], skip_special_tokens=True))

