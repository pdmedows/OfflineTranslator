import pyaudio

p = pyaudio.PyAudio()

print("Available audio devices:")
for i in range(p.get_device_count()):
    device_info = p.get_device_info_by_index(i)
    device_name = device_info['name']
    print(f"Device {i}: {device_name}")

p.terminate()
