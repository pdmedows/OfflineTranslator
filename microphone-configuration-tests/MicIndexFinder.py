import speech_recognition as sr

# Get the list of available microphones
available_microphones = sr.Microphone.list_microphone_names()

# Desired microphone name
desired_name = "USB PnP Audio Device: Audio (hw:4,0)"

# Initialize the microphone index to None (not found)
mic_index = None

# Find the index of the microphone with the desired name
for index, name in enumerate(available_microphones):
    if desired_name in name:
        mic_index = index
        break

if mic_index is not None:
    print(f"Microphone with name \"{desired_name}\" found at index {mic_index}")
else:
    print(f"Microphone with name \"{desired_name}\" not found in the list of available microphones.")
