# This dictionary includes all the MBart 50 language codes and detected languages.
combined_languages = {
    "Arabic": "ar_AR",
    "Czech": "cs_CZ",
    "German": "de_DE",
    "English": "en_XX",
    "Spanish": "es_XX",
    "Estonian": "et_EE",
    "Finnish": "fi_FI",
    "French": "fr_XX",
    "Gujarati": "gu_IN",
    "Hindi": "hi_IN",
    "Italian": "it_IT",
    "Japanese": "ja_XX",
    "Kazakh": "kk_KZ",
    "Korean": "ko_KR",
    "Lithuanian": "lt_LT",
    "Latvian": "lv_LV",
    "Burmese": "my_MM",
    "Nepali": "ne_NP",
    "Dutch": "nl_XX",
    "Romanian": "ro_RO",
    "Russian": "ru_RU",
    "Sinhala": "si_LK",
    "Turkish": "tr_TR",
    "Vietnamese": "vi_VN",
    "Chinese": "zh_CN",
    "Afrikaans": "af_ZA",
    "Azerbaijani": "az_AZ",
    "Bengali": "bn_IN",
    "Persian": "fa_IR",
    "Hebrew": "he_IL",
    "Croatian": "hr_HR",
    "Indonesian": "id_ID",
    "Georgian": "ka_GE",
    "Khmer": "km_KH",
    "Macedonian": "mk_MK",
    "Malayalam": "ml_IN",
    "Mongolian": "mn_MN",
    "Marathi": "mr_IN",
    "Polish": "pl_PL",
    "Pashto": "ps_AF",
    "Portuguese": "pt_XX",
    "Swedish": "sv_SE",
    "Swahili": "sw_KE",
    "Tamil": "ta_IN",
    "Telugu": "te_IN",
    "Thai": "th_TH",
    "Tagalog": "tl_XX",
    "Ukrainian": "uk_UA",
    "Urdu": "ur_PK",
    "Xhosa": "xh_ZA",
    "Galician": "gl_ES",
    "Slovenian": "sl_SI"
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

# load Speaker 1's audio and pad/trim it to fit 30 seconds
inputEs = whisper.load_audio("SpanishPodcast1.mp3")
inputEs = whisper.pad_or_trim(inputEs)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(inputEs).to(model.device)

# detect the spoken language
_, probs = model.detect_language(mel)
print(BarkLanguages[f"{max(probs, key=probs.get)}"])
rawSpeaker1Language = f"{max(probs, key=probs.get)}"
speaker1Language = BarkLanguages[f"{max(probs, key=probs.get)}"]
speaker1BarkLanguage = whisperToBarkLanguages[rawSpeaker1Language]

# decode the audio
options = whisper.DecodingOptions()
result = whisper.decode(model, mel, options)

# print the recognized text
print(result.text)
mBartInput1 = combined_languages[speaker1Language]
# translate Speaker 1's language to English first as a default, and then afterwards find Speaker 2's language from his or her respective input
tokenizer.src_lang = mBartInput1   # Enter input language code here
encoded_ar = tokenizer(result.text, return_tensors="pt")
generated_tokens = bartModel.generate(
    **encoded_ar,
    forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"]  # Enter output language code here
)
translation_es_en = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
print("Translation Spanish to English:", translation_es_en)

silence = np.zeros(int(0.20 * SAMPLE_RATE))  # fifth second of silence




text_prompt1 = translation_es_en[0]
text_prompt1.replace("\n", " ").strip()
# We split longer text into sentences using nltk and generate the sentences one by one.
sentences1 = nltk.sent_tokenize(text_prompt1)

speaker2BarkLanguage = "v2/en_speaker_6"
silence = np.zeros(int(0.20 * SAMPLE_RATE))  # fifth second of silence

pieces = []
for sentence in sentences1:
    print(sentence)
    audio_array = generate_audio(sentence, history_prompt=speaker2BarkLanguage)
    pieces += [audio_array, silence.copy()]
concatenated_audio = np.concatenate(pieces)

# save audio to disk
output_wav_path = "EnglishOutput1.wav"
write_wav(output_wav_path, SAMPLE_RATE, concatenated_audio)


# load Speaker 2's audio and pad/trim it to fit 30 seconds
inputEn = whisper.load_audio("EnglishInput2.mp3")
inputEn = whisper.pad_or_trim(inputEn)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(inputEn).to(model.device)

# detect the spoken language
_, probs = model.detect_language(mel)
print(BarkLanguages[f"{max(probs, key=probs.get)}"])
rawSpeaker2Language = f"{max(probs, key=probs.get)}"
speaker2Language = BarkLanguages[f"{max(probs, key=probs.get)}"]
speaker2BarkLanguage = whisperToBarkLanguages[rawSpeaker1Language]



# decode the audio
options = whisper.DecodingOptions()
result = whisper.decode(model, mel, options)

# print the recognized text
print(result.text)
mBartInput2 = combined_languages[speaker2Language]
# translate Speaker 2's language to Speaker1's Language
tokenizer.src_lang = mBartInput2   # Enter input language code here
encoded_ar = tokenizer(result.text, return_tensors="pt")
generated_tokens = bartModel.generate(
    **encoded_ar,
    forced_bos_token_id=tokenizer.lang_code_to_id[mBartInput1]   # Enter output language code here
      )
translation_en_es = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
print("Translation Englsih to Spanish:", translation_en_es)

silence = np.zeros(int(0.20 * SAMPLE_RATE))  # fifth second of silence




text_prompt2 = translation_en_es[0]
text_prompt2.replace("\n", " ").strip()
# We split longer text into sentences using nltk and generate the sentences one by one.
sentences2 = nltk.sent_tokenize(text_prompt2)

   
silence = np.zeros(int(0.20 * SAMPLE_RATE))  # fifth second of silence

pieces = []
for sentence in sentences2:
    print(sentence)
    audio_array = generate_audio(sentence, history_prompt=speaker1BarkLanguage)    #use Speaker 1's language to generate the audio for Speaker 1

    pieces += [audio_array, silence.copy()]
concatenated_audio = np.concatenate(pieces)

# save audio to disk
output_wav_path = "SpanishOutput1.wav"
write_wav(output_wav_path, SAMPLE_RATE, concatenated_audio)