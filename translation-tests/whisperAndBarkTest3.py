# This dictionary includes all the MBart 50 language codes and detected languages.
combined_languages = {
    "ar_AR": "Arabic",
    "cs_CZ": "Czech",
    "de_DE": "German",
    "en_XX": "English",
    "es_XX": "Spanish",
    "et_EE": "Estonian",
    "fi_FI": "Finnish",
    "fr_XX": "French",
    "gu_IN": "Gujarati",
    "hi_IN": "Hindi",
    "it_IT": "Italian",
    "ja_XX": "Japanese",
    "kk_KZ": "Kazakh",
    "ko_KR": "Korean",
    "lt_LT": "Lithuanian",
    "lv_LV": "Latvian",
    "my_MM": "Burmese",
    "ne_NP": "Nepali",
    "nl_XX": "Dutch",
    "ro_RO": "Romanian",
    "ru_RU": "Russian",
    "si_LK": "Sinhala",
    "tr_TR": "Turkish",
    "vi_VN": "Vietnamese",
    "zh_CN": "Chinese (Simplified)",
    "af_ZA": "Afrikaans",
    "az_AZ": "Azerbaijani",
    "bn_IN": "Bengali",
    "fa_IR": "Persian",
    "he_IL": "Hebrew",
    "hr_HR": "Croatian",
    "id_ID": "Indonesian",
    "ka_GE": "Georgian",
    "km_KH": "Khmer",
    "mk_MK": "Macedonian",
    "ml_IN": "Malayalam",
    "mn_MN": "Mongolian",
    "mr_IN": "Marathi",
    "pl_PL": "Polish",
    "ps_AF": "Pashto",
    "pt_XX": "Portuguese",
    "sv_SE": "Swedish",
    "sw_KE": "Swahili",
    "ta_IN": "Tamil",
    "te_IN": "Telugu",
    "th_TH": "Thai",
    "tl_XX": "Tagalog",
    "uk_UA": "Ukrainian",
    "ur_PK": "Urdu",
    "xh_ZA": "Xhosa",
    "gl_ES": "Galician",
    "sl_SI": "Slovenian"
}
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
    "uk": "Ukrainian",
    "el": "Greek",
    "ms": "Malay",
    "cs": "Czech",
    "ro": "Romanian",
    "da": "Danish",
    "hu": "Hungarian",
    "ta": "Tamil",
    "no": "Norwegian",
    "th": "Thai",
    "ur": "Urdu",
    "hr": "Croatian",
    "bg": "Bulgarian",
    "lt": "Lithuanian",
    "la": "Latin",
    "mi": "Maori",
    "ml": "Malayalam",
    "cy": "Welsh",
    "sk": "Slovak",
    "te": "Telugu",
    "fa": "Persian",
    "lv": "Latvian",
    "bn": "Bengali",
    "sr": "Serbian",
    "az": "Azerbaijani",
    "sl": "Slovenian",
    "kn": "Kannada",
    "et": "Estonian",
    "mk": "Macedonian",
    "br": "Breton",
    "eu": "Basque",
    "is": "Icelandic",
    "hy": "Armenian",
    "ne": "Nepali",
    "mn": "Mongolian",
    "bs": "Bosnian",
    "kk": "Kazakh",
    "sq": "Albanian",
    "sw": "Swahili",
    "gl": "Galician",
    "mr": "Marathi",
    "pa": "Punjabi",
    "si": "Sinhala",
    "km": "Khmer",
    "sn": "Shona",
    "yo": "Yoruba",
    "so": "Somali",
    "af": "Afrikaans",
    "oc": "Occitan",
    "ka": "Georgian",
    "be": "Belarusian",
    "tg": "Tajik",
    "sd": "Sindhi",
    "gu": "Gujarati",
    "am": "Amharic",
    "yi": "Yiddish",
    "lo": "Lao",
    "uz": "Uzbek",
    "fo": "Faroese",
    "ht": "Haitian Creole",
    "ps": "Pashto",
    "tk": "Turkmen",
    "nn": "Nynorsk",
    "mt": "Maltese",
    "sa": "Sanskrit",
    "lb": "Luxembourgish",
    "my": "Myanmar",
    "bo": "Tibetan",
    "tl": "Tagalog",
    "mg": "Malagasy",
    "as": "Assamese",
    "tt": "Tatar",
    "haw": "Hawaiian",
    "ln": "Lingala",
    "ha": "Hausa",
    "ba": "Bashkir",
    "jw": "Javanese",
    "su": "Sundanese",
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


from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

bartModel = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
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
model = whisper.load_model("base")

# load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio("SpanishPodcast1.mp3")
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio).to(model.device)

# detect the spoken language
_, probs = model.detect_language(mel)
print(BarkLanguages[f"{max(probs, key=probs.get)}"])
#barkSpeaker = whisperToBarkLanguages[f"{max(probs, key=probs.get)}"]
# decode the audio
options = whisper.DecodingOptions()
result = whisper.decode(model, mel, options)

# print the recognized text
print(result.text)

# translate Spanish to English
tokenizer.src_lang = "es_XX"
encoded_ar = tokenizer(result.text, return_tensors="pt")
generated_tokens = bartModel.generate(
    **encoded_ar,
    forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"]
)
translation_es_en = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
print("Translation Spanish to English:", translation_es_en)
SPEAKER = "v2/en_speaker_6"
silence = np.zeros(int(0.20 * SAMPLE_RATE))  # fifth second of silence




text_prompt = translation_es_en[0]
text_prompt.replace("\n", " ").strip()
# We split longer text into sentences using nltk and generate the sentences one by one.
sentences = nltk.sent_tokenize(text_prompt)

SPEAKER = "v2/en_speaker_6"
silence = np.zeros(int(0.20 * SAMPLE_RATE))  # fifth second of silence

pieces = []
for sentence in sentences:
    print(sentence)
    audio_array = generate_audio(sentence, history_prompt=SPEAKER)
    pieces += [audio_array, silence.copy()]
concatenated_audio = np.concatenate(pieces)

# save audio to disk
output_wav_path = "EnglishOutput1.wav"
write_wav(output_wav_path, SAMPLE_RATE, concatenated_audio)