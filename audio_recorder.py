import pyaudio
import numpy as np
import threading
import time
from queue import Queue
class AudioRecorder:
    def __init__(self, callback=None, rate=44100, chunk_size=1024, channels=1, device_index=None):
        self.rate = rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.device_index = device_index
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        self.callback = callback
        self.audio_queue = Queue()
        self.threshold = 0.02
        self.silence_timeout = 0.3
        self.recording_thread = None
        self.processing_thread = None
    def start_recording(self):
        if self.is_recording:
            return
        self.is_recording = True
        try:
            self.stream = self.p.open(
                format=pyaudio.paFloat32,
                channels=self.channels,
                rate=self.rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk_size
            )
            self.recording_thread = threading.Thread(target=self._record)
            self.recording_thread.daemon = True
            self.recording_thread.start()
            self.processing_thread = threading.Thread(target=self._process_audio)
            self.processing_thread.daemon = True
            self.processing_thread.start()
        except Exception as e:
            print(f"录音错误: {e}")
            self.is_recording = False
    def stop_recording(self):
        self.is_recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
    def _record(self):
        while self.is_recording:
            try:
                data = self.stream.read(self.chunk_size)
                audio_data = np.frombuffer(data, dtype=np.float32)
                self.audio_queue.put(audio_data)
                if self.callback:
                    self.callback(audio_data)
            except Exception as e:
                print(f"录音错误: {e}")
                time.sleep(0.1)
    def _process_audio(self):
        is_key_pressed = False
        key_audio = []
        last_sound_time = time.time()
        silence_duration = 0
        while self.is_recording:
            if not self.audio_queue.empty():
                audio_data = self.audio_queue.get()
                rms = np.sqrt(np.mean(np.square(audio_data)))
                if rms > self.threshold:
                    last_sound_time = time.time()
                    silence_duration = 0
                    if not is_key_pressed:
                        is_key_pressed = True
                        key_audio = [audio_data]
                        print(f"检测到可能的按键声音，音量: {rms:.6f}")
                    else:
                        key_audio.append(audio_data)
                else:
                    if is_key_pressed:
                        silence_duration = time.time() - last_sound_time
                        if silence_duration > self.silence_timeout:
                            is_key_pressed = False
                            if key_audio and len(key_audio) > 2:
                                full_audio = np.concatenate(key_audio)
                                print(f"按键事件结束，音频长度: {len(full_audio)}")
                                if self.callback:
                                    self.callback(full_audio, is_key_event=True)
                            key_audio = []
            else:
                time.sleep(0.01)
    def set_threshold(self, value):
        self.threshold = value
    def __del__(self):
        self.stop_recording()
        self.p.terminate()