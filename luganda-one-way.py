from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# pipeline approach
model_name = "Helsinki-NLP/opus-mt-en-lg"
translator = pipeline("translation", model=model_name)
translator("Hi my name is Daniel")

# loading model directly:
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("mps")
inputs = tokenizer("Hi my name is Daniel", return_tensors="pt").to("mps")
output = model.generate(**inputs)
translated_output = tokenizer.decode(output[0], skip_special_tokens=True)