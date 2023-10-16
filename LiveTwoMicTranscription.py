import argparse
import io
import os
import speech_recognition as sr    
import whisper
import torch
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
                        help="Don't use the English model.")
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for the mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=2,
                        help="How real-time the recording is in seconds.", type float)
    parser.add_argument("--phrase_timeout", default=3,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)
    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse',
                            help="Default microphone name for SpeechRecognition. "
                                 "Run this with 'list' to view available microphones.", type=str)
    args = parser.parse_args()

    # The last time a recording was retrieved from the queue.
    phrase_time = None
    # Current raw audio bytes.
    last_sample1 = bytes()
    last_sample2 = bytes()
    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue1 = Queue()
    data_queue2 = Queue()
    # We use SpeechRecognizer to record our audio because it has a nice feature where it can detect when speech ends.
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    # Definitely do this; dynamic energy compensation lowers the energy threshold dramatically to a point where the SpeechRecognizer never stops recording.
    recorder.dynamic_energy_threshold = False

    # Important for Linux users.
    # Prevents permanent application hang and crash by using the wrong microphone
    if 'linux' in platform:
        mic_name = args.default_microphone
        if not mic_name or mic_name == 'list':
            print("Available microphone devices are:")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"Microphone with name \"{name}\" found")
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source1 = sr.Microphone(sample_rate=16000, device_index=index)
                    # Set up a second microphone
                    source2 = sr.Microphone(sample_rate=16000, device_index=your_index)  # Replace 'your_index' with the correct index
                    break
    else:
        source1 = sr.Microphone(sample_rate=16000)
        # Set up a second microphone
        source2 = sr.Microphone(sample_rate=16000)  # Configure this line for the second microphone

    # Load / Download the model
    model = args.model
    if args.model != "large" and not args.non_english:
        model = model + ".en"
    audio_model = whisper.load_model(model)

    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout

    temp_file1 = NamedTemporaryFile().name
    temp_file2 = NamedTemporaryFile().name
    transcription1 = ['']
    transcription2 = ['']

    with source1:
        recorder.adjust_for_ambient_noise(source1)
    with source2:
        recorder.adjust_for_ambient_noise(source2)

    def record_callback(_, audio: sr.AudioData) -> None:
        """
        Threaded callback function to receive audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        # Determine the source of the audio and push it into the appropriate thread-safe queue.
        if audio.source == source1:
            data_queue1.put(audio.get_raw_data())
        elif audio.source == source2:
            data_queue2.put(audio.get_raw_data())

    # Create a background thread that will pass us raw audio bytes for the first microphone
    recorder.listen_in_background(source1, record_callback, phrase_time_limit=record_timeout)
    # Create a background thread for the second microphone
    recorder.listen_in_background(source2, record_callback, phrase_time_limit=record_timeout)

    # Cue the user that we're ready to go.
    print("Model loaded.\n")

    while True:
        try:
            now = datetime.utcnow()
            # Pull raw recorded audio from the queues for the first microphone
            if not data_queue1.empty():
                phrase_complete = False
                # If enough time has passed between recordings, consider the phrase complete.
                # Clear the current working audio buffer to start over with the new data.
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    last_sample1 = bytes()
                    phrase_complete = True
                # This is the last time we received new audio data from the queue.
                phrase_time = now

                # Concatenate our current audio data with the latest audio data for the first microphone
                while not data_queue1.empty():
                    data = data_queue1.get()
                    last_sample1 += data

                # Pull raw recorded audio from the queue for the second microphone
                if not data_queue2.empty():
                    phrase_complete = False
                    if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                        last_sample2 = bytes()
                        phrase_complete = True
                    phrase_time = now

                # Concatenate our current audio data with the latest audio data for the second microphone
                while not data_queue2.empty():
                    data = data_queue2.get()
                    last_sample2 += data

                # Use AudioData to convert the raw data to wav data for the first microphone
                audio_data1 = sr.AudioData(last_sample1, source1.SAMPLE_RATE, source1.SAMPLE_WIDTH)
                wav_data1 = io.BytesIO(audio_data1.get_wav_data())

                # Use AudioData to convert the raw data to wav data for the second microphone
                audio_data2 = sr.AudioData(last_sample2, source2.SAMPLE_RATE, source2.SAMPLE_WIDTH)
                wav_data2 = io.BytesIO(audio_data2.get_wav_data())

                # Write wav data to the temporary files as bytes for the first microphone
                with open(temp_file1, 'w+b') as f:
                    f.write(wav_data1.read())

                # Write wav data to the temporary files as bytes for the second microphone
                with open(temp_file2, 'w+b') as f:
                    f.write(wav_data2.read())

                # Read the transcription for the first microphone
                result1 = audio_model.transcribe(temp_file1, fp16=torch.cuda.is_available())
                text1 = result1['text'].strip()

                # Read the transcription for the second microphone
                result2 = audio_model.transcribe(temp_file2, fp16=torch.cuda.is_available())
                text2 = result2['text'].strip()

                # If we detected a pause between recordings, add a new item to our transcription for the first microphone
                if phrase_complete:
                    transcription1.append(text1)
                else:
                    transcription1[-1] = text1

                # If we detected a pause between recordings, add a new item to our transcription for the second microphone
                if phrase_complete:
                    transcription2.append(text2)
                else:
                    transcription2[-1] = text2

                # Clear the console to reprint the updated transcription for the first microphone
                os.system('cls' if os.name == 'nt' else 'clear')
                for line in transcription1:
                    print(f"Microphone 1: {line}")

                # Clear the console to reprint the updated transcription for the second microphone
                for line in transcription2:
                    print(f"Microphone 2: {line}")

                # Flush stdout
                print('', end='', flush=True)

                # Infinite loops are bad for processors; must sleep.
                sleep(0.25)
        except KeyboardInterrupt:
            break

    print("\n\nTranscription for Microphone 1:")
    for line in transcription1:
        print(line)

    print("\n\nTranscription for Microphone 2:")
    for line in transcription2:

        print(line)

if __name__ == "__main__":
    main()
