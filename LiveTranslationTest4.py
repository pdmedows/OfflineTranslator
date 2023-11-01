
import argparse
import io
import os
from datetime import datetime, timedelta
from queue import Queue
from sys import platform
from tempfile import NamedTemporaryFile
from time import sleep

import nltk  # we'll use this to split into sentences
import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd
import speech_recognition as sr
import torch
import whisper
from bark import SAMPLE_RATE, generate_audio
from bark.generation import generate_text_semantic, preload_models
from IPython.display import Audio
from lingua import Language, LanguageDetectorBuilder
from scipy.io.wavfile import write as write_wav
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration

# Bark requieres 12GB of VRAM, so we need to offload it to the CPU
os.environ["SUNO_OFFLOAD_CPU"] = "True"
os.environ["SUNO_USE_SMALL_MODELS"] = "True"

bartModel = MBartForConditionalGeneration.from_pretrained(
    "facebook/mbart-large-50-many-to-many-mmt"
)
tokenizer = MBart50TokenizerFast.from_pretrained(
    "facebook/mbart-large-50-many-to-many-mmt"
)


# This dictionary includes all the MBart 50 language codes and detected languages.
lingua_to_mbart_languages = {
    "Language.ARABIC": "ar_AR",
    "Language.CZECH": "cs_CZ",
    "Language.GERMAN": "de_DE",
    "Language.ENGLISH": "en_XX",
    "Language.SPANISH": "es_XX",
    "Language.ESTONIAN": "et_EE",
    "Language.FINNISH": "fi_FI",
    "Language.FRENCH": "fr_XX",
    "Language.GUJARATI": "gu_IN",
    "Language.HINDI": "hi_IN",
    "Language.ITALIAN": "it_IT",
    "Language.JAPANESE": "ja_XX",
    "Language.KAZAKH": "kk_KZ",
    "Language.KOREAN": "ko_KR",
    "Language.LITHUANIAN": "lt_LT",
    "Language.LATVIAN": "lv_LV",
    "Language.BURMESE": "my_MM",
    "Language.NEPALI": "ne_NP",
    "Language.DUTCH": "nl_XX",
    "Language.ROMANIAN": "ro_RO",
    "Language.RUSSIAN": "ru_RU",
    "Language.SINHALA": "si_LK",
    "Language.TURKISH": "tr_TR",
    "Language.VIETNAMESE": "vi_VN",
    "Language.CHINESE": "zh_CN",
    "Language.AFRIKAANS": "af_ZA",
    "Language.AZERBAIJANI": "az_AZ",
    "Language.BENGALI": "bn_IN",
    "Language.PERSIAN": "fa_IR",
    "Language.HEBREW": "he_IL",
    "Language.CROATIAN": "hr_HR",
    "Language.INDONESIAN": "id_ID",
    "Language.GEORGIAN": "ka_GE",
    "Language.KHMER": "km_KH",
    "Language.MACEDONIAN": "mk_MK",
    "Language.MALAYALAM": "ml_IN",
    "Language.MONGOLIAN": "mn_MN",
    "Language.MARATHI": "mr_IN",
    "Language.POLISH": "pl_PL",
    "Language.PASHTO": "ps_AF",
    "Language.PORTUGUESE": "pt_XX",
    "Language.SWEDISH": "sv_SE",
    "Language.SWAHILI": "sw_KE",
    "Language.TAMIL": "ta_IN",
    "Language.TELUGU": "te_IN",
    "Language.THAI": "th_TH",
    "Language.TAGALOG": "tl_XX",
    "Language.UKRAINIAN": "uk_UA",
    "Language.URDU": "ur_PK",
    "Language.XHOSA": "xh_ZA",
    "Language.GALICIAN": "gl_ES",
    "Language.SLOVENIAN": "sl_SI",
}

lingua_to_bark_languages = {
    "Language.ENGLISH": "v2/en_speaker_6",
    "Language.CHINESE": "v2/zh_speaker_1",
    "Language.GERMAN": "v2/de_speaker_2",
    "Language.SPANISH": "v2/es_speaker_1",
    "Language.RUSSIAN": "v2/ru_speaker_4",
    "Language.KOREAN": "v2/ko_speaker_4",
    "Language.FRENCH": "v2/fr_speaker_3",
    "Language.JAPANESE": "v2/ja_speaker_6",
    "Language.PORTUGESE": "v2/pt_speakpter_3",
    "Language.TURKISH": "v2/tr_speaker_2",
    "Language.POLISH": "v2/pl_speaker_7",
    "Language.ITALIAN": "v2/it_speaker_4",
    "Language.HINDI": "v2/hi_speaker_6",
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="medium",
        help="Model to use",
        choices=["tiny", "base", "small", "medium", "large"],
    )
    parser.add_argument(
        "--non_english", action="store_true", help="Don't use the english model."
    )
    parser.add_argument(
        "--energy_threshold",
        default=1000,
        help="Energy level for mic to detect.",
        type=int,
    )
    parser.add_argument(
        "--record_timeout",
        default=2,
        help="How real time the recording is in seconds.",
        type=float,
    )
    parser.add_argument(
        "--phrase_timeout",
        default=3,
        help="How much empty space between recordings before we "
        "consider it a new line in the transcription.",
        type=float,
    )
    if "linux" in platform:
        parser.add_argument(
            "--default_microphone",
            default="pulse",
            help="Default microphone name for SpeechRecognition. "
            "Run this with 'list' to view available Microphones.",
            type=str,
        )
        parser.add_argument(
            "--default_microphone2",
            default="pulse",
            help="Default microphone name for SpeechRecognition. "
            "Run this with 'list' to view available Microphones.",
            type=str,
        )
    args = parser.parse_args()

    # The last time a recording was retrieved from the queue.
    phrase_time = None
    # Current raw audio bytes.
    last_sample = bytes()
    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = Queue()
    # We use SpeechRecognizer to record our audio because it has a nice feature where it can detect when speech ends.
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    # Definitely do this, dynamic energy compensation lowers the energy threshold dramatically to a point where the SpeechRecognizer never stops recording.
    recorder.dynamic_energy_threshold = False
    detected_language1 = Language.ENGLISH  # Set placeholder value to English
    detected_language2 = Language.ENGLISH

    # Important for linux users.
    # Prevents permanent application hang and crash by using the wrong Microphone
    if "linux" in platform:
        mic_name = args.default_microphone
        if not mic_name or mic_name == "list":
            print("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f'Microphone with name "{name}" found')
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source1 = sr.Microphone(sample_rate=16000, device_index=index)
                    source2 = sr.Microphone(sample_rate=16000, device_index=index)
                    break
    else:
        source1 = sr.Microphone(sample_rate=16000)
        source2 = sr.Microphone(sample_rate=16000)
    # Load / Download model
    # Include all languages available in the library.
    detector = LanguageDetectorBuilder.from_all_spoken_languages().build()
    model = args.model
    if args.model != "large" and not args.non_english:
        model = model + ".en"
    audio_model = whisper.load_model(model)

    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout

    temp_file = NamedTemporaryFile().name

    # Initialize the Recognizer
    recognizer = sr.Recognizer()
    with source1 as mic1, source2 as mic2:
        recognizer.adjust_for_ambient_noise(mic1)
        recognizer.adjust_for_ambient_noise(mic2)

    def play_audio(file_path):
        # Read the WAV file
        sample_rate, audio_data = wav.read(file_path)

        # Ensure the audio data is in floating-point format
        audio_data = audio_data.astype(np.float32)

        # Play the audio
        sd.play(audio_data, sample_rate)
        sd.wait()

    def record_callback(_, audio: sr.AudioData) -> None:
        """
        Threaded callback function to receive audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        # Grab the raw bytes and push it into the thread safe queue.
        data = audio.get_raw_data()
        data_queue.put(data)

    # Create a background thread that will pass us raw audio bytes.
    # We could do this manually but SpeechRecognizer provides a nice helper.
    recorder.listen_in_background(
        source1, record_callback, phrase_time_limit=record_timeout
    )
    recorder.listen_in_background(
        source2, record_callback, phrase_time_limit=record_timeout
    )

    # Cue the user that we're ready to go.
    print("Model loaded.\n")
    translation_to_speaker1 = [
        ""
    ]  # Define and initialize the variable outside the if block
    translation_to_speaker2 = [
        ""
    ]  # Define and initialize the variable outside the if block
    transcription = [""]
    while True:
        try:
            now = datetime.utcnow()
            # Pull raw recorded audio from the queue.
            if not data_queue.empty():
                phrase_complete = False
                # If enough time has passed between recordings, consider the phrase complete.
                # Clear the current working audio buffer to start over with the new data.
                if phrase_time and now - phrase_time > timedelta(
                    seconds=phrase_timeout
                ):
                    last_sample = bytes()
                    phrase_complete = True
                # This is the last time we received new audio data from the queue.
                phrase_time = now

                # Concatenate our current audio data with the latest audio data.
                while not data_queue.empty():
                    data = data_queue.get()
                    last_sample += data

                # Use AudioData to convert the raw data to wav data.
                audio_data = sr.AudioData(
                    last_sample, source1.SAMPLE_RATE, source1.SAMPLE_WIDTH
                )
                wav_data = io.BytesIO(audio_data.get_wav_data())

                # Write wav data to the temporary file as bytes.
                with open(temp_file, "w+b") as f:
                    f.write(wav_data.read())

                # Read the transcription.
                result = audio_model.transcribe(
                    temp_file, fp16=torch.cuda.is_available()
                )
                text = result["text"].strip()
                if text:
                    # Detect the language of the transcribed text
                    detected_language1 = detector.detect_language_of(text)
                else:
                    detected_language1 = "Unknown"  # Handle empty text

                # If we detected a pause between recordings, add a new item to our transcription.
                # Otherwise edit the existing one.
                if phrase_complete:
                    transcription.append(text)
                else:
                    transcription[-1] = text
                # If you want to add the detected language to the output, you can modify the printing code
                print(f"(Detected Language 1: {detected_language1})")
                print(f" (Detected Language 2: {detected_language2})")

                translation_to_speaker2 = [""]
                # Ensure that the detected language is a valid key in the dictionary
                if (
                    repr(detected_language1) in lingua_to_mbart_languages
                    and repr(detected_language2) in lingua_to_mbart_languages
                ):
                    mBartInput = lingua_to_mbart_languages[repr(detected_language1)]
                    print("MBart Input Language:", mBartInput)  # Debug print statement
                    # translate Spanish to English
                    tokenizer.src_lang = mBartInput
                    encoded_text = tokenizer(text, return_tensors="pt")
                    generated_tokens = bartModel.generate(
                        **encoded_text,
                        forced_bos_token_id=tokenizer.lang_code_to_id[
                            lingua_to_mbart_languages[repr(detected_language2)]
                        ],  # Translate to Speaker 1's language
                    )
                    translation_to_speaker2 = tokenizer.batch_decode(
                        generated_tokens, skip_special_tokens=True
                    )

                    print("Translation to Speaker 2: ", translation_to_speaker2)
                else:
                    print(
                        "Detected language not found in dictionary:", detected_language2
                    )
                    translation_to_speaker2 = [""]
                # Clear the console to reprint the updated transcription.
                os.system("cls" if os.name == "nt" else "clear")

                for line in translation_to_speaker2:
                    print(line)
                if (
                    translation_to_speaker2 != [""]
                    and repr(detected_language2) in lingua_to_bark_languages
                ):  # Check to see if the audio can be generated using Bark
                    text_prompt1 = translation_to_speaker2[0]

                    text_prompt1.replace("\n", " ").strip()
                    # We split longer text into sentences using nltk and generate the sentences one by one.
                    sentences1 = nltk.sent_tokenize(text_prompt1)

                    speaker_bark_language2 = lingua_to_bark_languages[
                        repr(detected_language2)
                    ]

                    silence = np.zeros(
                        int(0.20 * SAMPLE_RATE)
                    )  # fifth second of silence

                    pieces = []
                    for sentence in sentences1:
                        print(sentence)
                        audio_array = generate_audio(
                            sentence, history_prompt=speaker_bark_language2
                        )
                        pieces += [audio_array, silence.copy()]
                    concatenated_audio = np.concatenate(pieces)

                    # save audio to disk
                    output_wav_path1 = "Speaker1Output.wav"
                    write_wav(output_wav_path1, SAMPLE_RATE, concatenated_audio)
                    # play audio
                    play_audio(output_wav_path1)

                # Flush stdout.
                print("", end="", flush=True)

                # Infinite loops are bad for processors, must sleep.
                sleep(0.25)

                # Now do the same for Speaker 2

                now = datetime.utcnow()
            # Pull raw recorded audio from the queue.
            if not data_queue.empty():
                phrase_complete = False
                # If enough time has passed between recordings, consider the phrase complete.
                # Clear the current working audio buffer to start over with the new data.
                if phrase_time and now - phrase_time > timedelta(
                    seconds=phrase_timeout
                ):
                    last_sample = bytes()
                    phrase_complete = True
                # This is the last time we received new audio data from the queue.
                phrase_time = now

                # Concatenate our current audio data with the latest audio data.
                while not data_queue.empty():
                    data = data_queue.get()
                    last_sample += data
                # Use AudioData to convert the raw data to wav data.
                audio_data = sr.AudioData(
                    last_sample, source1.SAMPLE_RATE, source1.SAMPLE_WIDTH
                )
                wav_data = io.BytesIO(audio_data.get_wav_data())

                # Write wav data to the temporary file as bytes.
                with open(temp_file, "w+b") as f:
                    f.write(wav_data.read())

                # Read the transcription.
                result = audio_model.transcribe(
                    temp_file, fp16=torch.cuda.is_available()
                )
                text = result["text"].strip()
                if text:
                    # Detect the language of the transcribed text
                    detected_language2 = detector.detect_language_of(text)
                else:
                    detected_language2 = "Unknown"  # Handle empty text

                # If we detected a pause between recordings, add a new item to our transcription.
                # Otherwise edit the existing one.
                if phrase_complete:
                    transcription.append(text)
                else:
                    transcription[-1] = text
                # If you want to add the detected language to the output, you can modify the printing code
                print(f"(Detected Language 1: {detected_language1})")
                print(f" (Detected Language 2: {detected_language2})")

                # Ensure that the detected language is a valid key in the dictionary
                if (
                    repr(detected_language2) in lingua_to_mbart_languages
                    and repr(detected_language1) in lingua_to_mbart_languages
                ):
                    mBartInput = lingua_to_mbart_languages[repr(detected_language1)]
                    print("MBart Input Language:", mBartInput)  # Debug print statement
                    # translate Spanish to English
                    tokenizer.src_lang = mBartInput
                    encoded_text = tokenizer(text, return_tensors="pt")
                    generated_tokens = bartModel.generate(
                        **encoded_text,
                        forced_bos_token_id=tokenizer.lang_code_to_id[
                            lingua_to_mbart_languages[repr(detected_language1)]
                        ],  # Translate to Speaker 1's language
                    )
                    translation_to_speaker1 = tokenizer.batch_decode(
                        generated_tokens, skip_special_tokens=True
                    )

                    print("Translation to Speaker 1:", translation_to_speaker1)
                else:
                    print(
                        "Detected language not found in dictionary:", detected_language1
                    )
                    translation_to_speaker1 = [""]
                # Clear the console to reprint the updated transcription.
                os.system("cls" if os.name == "nt" else "clear")

                for line in translation_to_speaker1:
                    print(line)
                if (
                    translation_to_speaker1 != [""]
                    and repr(detected_language1) in lingua_to_bark_languages
                ):  # Translate to Speaker 1' sound if possible
                    text_prompt2 = translation_to_speaker1[0]

                    text_prompt2.replace("\n", " ").strip()
                    # We split longer text into sentences using nltk and generate the sentences one by one.
                    sentences2 = nltk.sent_tokenize(text_prompt2)

                    speaker_bark_language1 = lingua_to_bark_languages[
                        repr(detected_language1)
                    ]
                    silence = np.zeros(
                        int(0.20 * SAMPLE_RATE)
                    )  # fifth second of silence

                    pieces = []
                    for sentence in sentences2:
                        print(sentence)
                        audio_array = generate_audio(
                            sentence, history_prompt=speaker_bark_language1
                        )
                        pieces += [audio_array, silence.copy()]
                    concatenated_audio = np.concatenate(pieces)

                    # save audio to disk
                    output_wav_path2 = "Speaker2Output.wav"
                    write_wav(output_wav_path2, SAMPLE_RATE, concatenated_audio)
                    # play audio
                    play_audio(output_wav_path2)
                    # Flush stdout.
                print("", end="", flush=True)

                # Infinite loops are bad for processors, must sleep.
                sleep(0.25)

        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
