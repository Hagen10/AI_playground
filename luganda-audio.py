import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

model_name = "birgermoell/wav2vec2-luganda"

processor = Wav2Vec2Processor.from_pretrained(model_name)
# "mps" is used for this mac (would have been "cuda" if there was a GPU)
model = Wav2Vec2ForCTC.from_pretrained(model_name).to("mps")

speech, sr = torchaudio.load("luganda_audio.wav")
speech = speech.squeeze().numpy()
# "mps" is used for this mac (would have been "cuda" if there was a GPU)
inputs = processor(speech, sampling_rate=sr, return_tensors="pt", padding="longest").to("mps")
logits = model(inputs.input_values).logits
predicted_ids = logits.argmax(dim=-1)

print(processor.decode(predicted_ids[0]))

