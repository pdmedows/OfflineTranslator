BarkLanguages = {
    "en": "English",
    "zh": "Chinese",
    "de": "German",
    "es": "Spanish",
    "ru": "Russian",
    "ko": "Korean",
    "fr": "French",
    "ja": "Japanese",
    "pt": "Portuguese",
    "tr": "Turkish",
    "pl": "Polish",
    "ca": "Catalan",
    "nl": "Dutch",
    "ar": "Arabic",
    "sv": "Swedish",
    "it": "Italian",
    "id": "Indonesian",
    "hi": "Hindi",
    "fi": "Finnish",
    "vi": "Vietnamese",
    "he": "Hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
}

whisperToBarkLanguages = {"en": "v2/en_speaker_6",
    "zh": "v2/zh_speaker_1",
    "de": "v2/de_speaker_2",
    "es": "v2/es_speaker_1",
    "ru": "v2/ru_speaker_4",
    "ko": "v2/ko_speaker_4",
    "fr": "v2/fr_speaker_3",
    "ja": "v2/ja_speaker_6",
    "pt": "v2/pt_speaker_3",
    "tr": "v2/tr_speaker_2",
    "pl": "v2/pl_speaker_7",
    "it": "v2/it_speaker_4",
    "hi": "v2/hi_speaker_6"
    }

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

import whisper
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

# Load the T5 model and tokenizer for translation
model_name = "t5-base"  # You can change this to a different T5 model
t5model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

#Create Translator Function
def translate_text(input_text, source_language="es", target_language="en"):  
    # Preprocess input text
    input_text = f"translate {source_language} to {target_language}: {input_text}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Generate translation
    translation = t5model.generate(input_ids, max_length=100, num_beams=4, early_stopping=True)

    # Decode and return the translated text
    translated_text = tokenizer.decode(translation[0], skip_special_tokens=True)
    return translated_text

model = whisper.load_model("base")
# Transcribe Speaker 1 audio
# load audio and pad/trim it to fit 30 seconds
audio1 = whisper.load_audio("SpanishPodcast1.mp3")
audio1 = whisper.pad_or_trim(audio1)

# make log-Mel spectrogram and move to the same device as the model
mel1 = whisper.log_mel_spectrogram(audio1).to(model.device)

# detect the spoken language
_, probs = model.detect_language(mel1)
print(f"Detected language: {max(probs, key=probs.get)}")
speaker1Language = f"{max(probs, key=probs.get)}"
barkSpeaker1 = whisperToBarkLanguages[f"{max(probs, key=probs.get)}"]
# decode the audio
options = whisper.DecodingOptions()
result = whisper.decode(model, mel1, options)

# print the recognized text
print(result.text)
print(speaker1Language)
print(BarkLanguages[speaker1Language])
#Translate Speaker 1 Transcription To English
translated_text = translate_text(result.text, speaker1Language, "en")
print(f"Translated text: {translated_text}")

#Geerate Tranlsated English Audio 

text_prompt = translated_text
text_prompt.replace("\n", " ").strip()
# We split longer text into sentences using nltk and generate the sentences one by one.
sentences = nltk.sent_tokenize(text_prompt)

SPEAKER1 = "v2/en_speaker_6"
silence = np.zeros(int(0.20 * SAMPLE_RATE))  # fifth second of silence

pieces = []
for sentence in sentences:
    audio_array = generate_audio(sentence, history_prompt=SPEAKER1)
    pieces += [audio_array, silence.copy()]
concatenated_audio1 = np.concatenate(pieces)

# save audio to disk
output_wav_path = "EnglishTranslation1Output.wav"
write_wav(output_wav_path, SAMPLE_RATE, concatenated_audio1)
  


# print the recognized text
print(translated_text)
