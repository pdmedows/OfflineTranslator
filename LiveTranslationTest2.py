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
    "Language.SLOVENIAN": "sl_SI"
}
translation_to_en = ['']  # Initialize an empty srting in a list for the translated text
import argparse
import io
import os
import speech_recognition as sr
import whisper
import torch
from lingua import Language, LanguageDetectorBuilder
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

bartModel = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")



from datetime import datetime, timedelta
from queue import Queue
from tempfile import NamedTemporaryFile
from time import sleep
from sys import platform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medium", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--non_english", action='store_true',
                        help="Don't use the english model.")
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=2,
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=3,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)
    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse',
                            help="Default microphone name for SpeechRecognition. "
                                 "Run this with 'list' to view available Microphones.", type=str)
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

    # Important for linux users.
    # Prevents permanent application hang and crash by using the wrong Microphone
    if 'linux' in platform:
        mic_name = args.default_microphone
        if not mic_name or mic_name == 'list':
            print("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"Microphone with name \"{name}\" found")
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = sr.Microphone(sample_rate=16000, device_index=index)
                    break
    else:
        source = sr.Microphone(sample_rate=16000)

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
    transcription = ['']

    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio:sr.AudioData) -> None:
        """
        Threaded callback function to receive audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        # Grab the raw bytes and push it into the thread safe queue.
        data = audio.get_raw_data()
        data_queue.put(data)

    # Create a background thread that will pass us raw audio bytes.
    # We could do this manually but SpeechRecognizer provides a nice helper.
    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    # Cue the user that we're ready to go.
    print("Model loaded.\n")

    while True:
        try:
            now = datetime.utcnow()
            # Pull raw recorded audio from the queue.
            if not data_queue.empty():
                phrase_complete = False
                # If enough time has passed between recordings, consider the phrase complete.
                # Clear the current working audio buffer to start over with the new data.
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    last_sample = bytes()
                    phrase_complete = True
                # This is the last time we received new audio data from the queue.
                phrase_time = now

                # Concatenate our current audio data with the latest audio data.
                while not data_queue.empty():
                    data = data_queue.get()
                    last_sample += data

                # Use AudioData to convert the raw data to wav data.
                audio_data = sr.AudioData(last_sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
                wav_data = io.BytesIO(audio_data.get_wav_data())

                # Write wav data to the temporary file as bytes.
                with open(temp_file, 'w+b') as f:
                    f.write(wav_data.read())

                # Read the transcription.
                result = audio_model.transcribe(temp_file, fp16=torch.cuda.is_available())
                text = result['text'].strip()
                if text: 
                # Detect the language of the transcribed text
                    detected_language = detector.detect_language_of(text)
                else:
                    detected_language = "Unknown"  # Handle empty text
                

                # If you want to add the detected language to the output, you can modify the printing code
                text_with_language = f"{text} (Language: {detected_language})"


                # If we detected a pause between recordings, add a new item to our transcription.
                # Otherwise edit the existing one.
                if phrase_complete:
                    transcription.append(text)
                else:
                    transcription[-1] = text
               
                # Ensure that the detected language is a valid key in the dictionary
                if repr(detected_language) in lingua_to_mbart_languages:
                    mBartInput = lingua_to_mbart_languages[repr(detected_language)]
                    print("MBart Input Language:", mBartInput)  # Debug print statement
                    # translate Spanish to English
                    tokenizer.src_lang = mBartInput  
                    encoded_text = tokenizer(text, return_tensors="pt")
                    generated_tokens = bartModel.generate(
                    **encoded_text,
                    forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"]
                    )
                    translation_to_en = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

                    print("Translation to English:", translation_to_en)
                else:
                    print("Detected language not found in dictionary:", detected_language)
                    translation_to_en = ''
                # Clear the console to reprint the updated transcription.
                os.system('cls' if os.name=='nt' else 'clear')
                for line in transcription:
                    print(line)
                for line in translation_to_en:
                    print(line)
                
                
                
                # Flush stdout.
                print('', end='', flush=True)
                

                # Infinite loops are bad for processors, must sleep.
                sleep(0.25)
        except KeyboardInterrupt:
            break

    print("\n\nTranscription:")
    for line in transcription:
        print(line)
    if translation_to_en:
        for line in translation_to_en:
            print(line)

   


if __name__ == "__main__":
    main()