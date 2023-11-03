import pyaudio

p = pyaudio.PyAudio()
import speech_recognition as sr
print("Available audio devices:")
for i in range(p.get_device_count()):
    device_info = p.get_device_info_by_index(i)
    device_name = device_info['name']
    print(f"Device {i}: {device_name}")


# Get the list of available microphones
available_microphones = sr.Microphone.list_microphone_names()
    
    # Desired microphone name
desired_name1 = "USB PnP Audio Device: Audio (hw:3,0)"

    # Initialize the microphone index to None (not found)
mic_index1 = None

    # Find the index of the microphone with the desired name
for index, name in enumerate(available_microphones):
    if desired_name1 in name:
        mic_index1 = index
        break
    # Desired microphone name
desired_name2 = "USB PnP Audio Device: Audio (hw:4,0)"

    # Initialize the microphone index to None (not found)
mic_index2 = None

    # Find the index of the microphone with the desired name
for index, name in enumerate(available_microphones):
    if desired_name2 in name:
        mic_index2 = index
        break
print(f"mic_index1: {mic_index1}")
print(f"mic_index2: {mic_index2}")

import speech_recognition as sr

# List available microphones
available_microphones = sr.Microphone.list_microphone_names()
print("Available microphones:")
for index, name in enumerate(available_microphones):
    print(f"Microphone {index}: {name}")

# Choose the desired microphone by name
desired_microphone_name = "USB PnP Audio Device: Audio (hw:3,0)"  # Replace with the name of the microphone you want to test

# Initialize the microphone with a specific sample rate
sample_rate_to_test = 16000  # Specify the sample rate to test
microphone = sr.Microphone(sample_rate=sample_rate_to_test, device_index=available_microphones.index(desired_microphone_name))

# Record audio
with microphone as source:
    print(f"Recording from {desired_microphone_name} at {sample_rate_to_test} Hz")
    recognizer = sr.Recognizer()
    audio = recognizer.listen(source)

# Check the actual sample rate of the recorded audio
actual_sample_rate = audio.sample_rate
print(f"Actual Sample Rate: {actual_sample_rate} Hz")


p.terminate()
