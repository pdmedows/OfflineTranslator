import speech_recognition as sr

# List available microphones
available_microphones = sr.Microphone.list_microphone_names()
print("Available microphones:")
for index, name in enumerate(available_microphones):
    print(f"Microphone {index}: {name}")

# Choose the desired microphone by name
desired_microphone_name = "USB PnP Audio Device: Audio (hw:4,0)"  # Replace with the name of the microphone you want to test

# Initialize the microphone with a specific sample rate
sample_rate_to_test = 16000  # Specify the sample rate to test
microphone = sr.Microphone(sample_rate=sample_rate_to_test, device_index=available_microphones.index(desired_microphone_name))

# Record audio within the with statement
with microphone as source:
    print(f"Recording from {desired_microphone_name} at {sample_rate_to_test} Hz")
    recognizer = sr.Recognizer()
    audio = recognizer.listen(source)

# Check the actual sample rate of the recorded audio
actual_sample_rate = audio.sample_rate
print(f"Actual Sample Rate: {actual_sample_rate} Hz")
