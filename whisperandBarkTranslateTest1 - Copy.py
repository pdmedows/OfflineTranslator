BarkLanguages = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
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
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load the T5 model and tokenizer
model_name = "t5-large"  # You can use other T5 variants like "t5-base" or "t5-large"
T5model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

def translate_text(input_text, target_language="translate English to French:"):
    # Preprocess input text by specifying the translation task and target language
    input_text = f"{target_language} {input_text}"
    
    # Tokenize input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True)

    # Generate translation
    translation_ids = T5model.generate(input_ids, max_length=100, num_beams=4, early_stopping=True)

    # Decode and return the translated text
    translated_text = tokenizer.decode(translation_ids[0], skip_special_tokens=True)
    return translated_text

model = whisper.load_model("base")

# load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio("SpanishPodcast1.mp3")
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio).to(model.device)

# detect the spoken language
_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")
barkSpeaker = whisperToBarkLanguages[f"{max(probs, key=probs.get)}"]
# decode the audio
options = whisper.DecodingOptions()
result = whisper.decode(model, mel, options)

# print the recognized text
print(result.text)

text_prompt = result.text
text_prompt.replace("\n", " ").strip()
# We split longer text into sentences using nltk and generate the sentences one by one.
sentences = nltk.sent_tokenize(text_prompt)
translated_text = []
for sentence in sentences:
    #translated_text += translate_text(sentence, "translate Spanish to English:")
    translated_text += sentence
translated_text = " ".join(translated_text)
SPEAKER = "v2/en_speaker_6"
silence = np.zeros(int(0.20 * SAMPLE_RATE))  # fifth second of silence
print(f"Translated text: {translated_text}")
pieces = []
for sentence in sentences:
    audio_array = generate_audio(sentence, history_prompt=SPEAKER)
    pieces += [audio_array, silence.copy()]
concatenated_audio = np.concatenate(pieces)




text_prompt = result.text
text_prompt.replace("\n", " ").strip()
# We split longer text into sentences using nltk and generate the sentences one by one.
sentences = nltk.sent_tokenize(text_prompt)

SPEAKER = "v2/en_speaker_6"
silence = np.zeros(int(0.20 * SAMPLE_RATE))  # fifth second of silence

pieces = []
for sentence in sentences:
    audio_array = generate_audio(sentence, history_prompt=SPEAKER)
    pieces += [audio_array, silence.copy()]
concatenated_audio = np.concatenate(pieces)

# save audio to disk
output_wav_path = "EnglishOutput1.wav"
write_wav(output_wav_path, SAMPLE_RATE, concatenated_audio)
  


# print the recognized text
print(result.text)
