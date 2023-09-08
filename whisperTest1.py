import os
from scipy.io.wavfile import write as write_wav
from IPython.display import Audio
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["SUNO_USE_SMALL_MODELS"] = "1"
os.environ["SUNO_OFFLOAD_CPU"] = "0"

import nltk  # we'll use this to split into sentences
from bark.generation import (
    generate_text_semantic,
    preload_models,
)
from bark import generate_audio, SAMPLE_RATE
import numpy as np
import torch
# torch.cuda.reset_peak_memory_stats()
preload_models()
import whisper

model = whisper.load_model("base")
result = model.transcribe("output_audio.wav")
print(result["text"])
text_prompt = result["text"]
text_prompt.replace("\n", " ").strip()
# We split longer text into sentences using nltk and generate the sentences one by one.
sentences = nltk.sent_tokenize(text_prompt)

SPEAKER = "v2/en_speaker_6"
silence = np.zeros(int(0.25 * SAMPLE_RATE))  # quarter second of silence

pieces = []
for sentence in sentences:
    audio_array = generate_audio(sentence, history_prompt=SPEAKER)
    pieces += [audio_array, silence.copy()]
concatenated_audio = np.concatenate(pieces)

# save audio to disk
output_wav_path = "whisper_and_bark_test1.wav"
write_wav(output_wav_path, SAMPLE_RATE, concatenated_audio)
  