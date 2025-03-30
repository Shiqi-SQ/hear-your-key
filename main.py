import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import Qt, QTimer
import pyaudio
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.family'] = 'sans-serif'
from ui import MainWindow
from audio_recorder import AudioRecorder
from feature_extractor import FeatureExtractor
from model import KeyboardModel
class KeyboardSoundApp:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.window = MainWindow()
        self.feature_extractor = FeatureExtractor()
        self.model = KeyboardModel()
        self.recorder = None
        self.init_microphones()
        self.connect_signals()
        self.init_recorder()
        self.update_sample_count()
        self.window.show()
    def init_microphones(self):
        p = pyaudio.PyAudio()
        self.window.mic_combo.clear()
        self.window.mic_combo.addItem("默认麦克风", -1)
        for i in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                name = device_info['name']
                self.window.mic_combo.addItem(name, i)
        p.terminate()
    def connect_signals(self):
        self.window.learn_mode_radio.toggled.connect(self.toggle_mode)
        self.window.match_mode_radio.toggled.connect(self.toggle_mode)
        self.window.mic_combo.currentIndexChanged.connect(self.change_microphone)
        self.window.sensitivity_slider.valueChanged.connect(self.change_sensitivity)
        self.window.add_sample_btn.clicked.connect(self.add_sample)
        self.window.train_model_btn.clicked.connect(self.train_model)
    def init_recorder(self):
        self.recorder = AudioRecorder(callback=self.process_audio)
        self.recorder.start_recording()
    def process_audio(self, audio_data, is_key_event=False):
        try:
            self.window.visualizer.update_waveform(audio_data)
            if len(audio_data) > 512:
                spec_data = self.feature_extractor.get_spectrogram(audio_data)
                self.window.visualizer.update_spectrogram(spec_data)
            if is_key_event:
                features = self.feature_extractor.extract_features(audio_data)
                if self.window.learn_mode_radio.isChecked():
                    self.window.log("检测到按键声音，请在输入框中输入对应的按键")
                    self.current_features = features
                else:
                    if self.model.is_trained:
                        key, confidence = self.model.predict(features)
                        if key:
                            self.window.update_result(key, confidence)
                            self.window.log(f"检测到按键: {key} (置信度: {confidence:.2f})")
                        else:
                            self.window.update_result("未识别")
                            self.window.log("无法识别按键")
                    else:
                            self.window.log("模型尚未训练，请先切换到学习模式训练模型")
        except Exception as e:
            print(f"音频处理错误: {e}")
    def toggle_mode(self, checked):
        if checked:
            if self.window.learn_mode_radio.isChecked():
                self.window.log("切换到学习模式")
            else:
                self.window.log("切换到匹配模式")
                if not self.model.is_trained:
                    self.window.log("警告：模型尚未训练，请先在学习模式下训练模型")
    def change_microphone(self, index):
        device_id = self.window.mic_combo.itemData(index)
        self.window.log(f"切换到麦克风: {self.window.mic_combo.currentText()}")
        if self.recorder:
            self.recorder.stop_recording()
        if device_id != -1:
            self.recorder = AudioRecorder(callback=self.process_audio, device_index=device_id)
        else:
            self.recorder = AudioRecorder(callback=self.process_audio)
        self.recorder.start_recording()
    def change_sensitivity(self, value):
        if self.recorder:
            threshold = 0.05 * (100 - value) / 100
            self.recorder.set_threshold(threshold)
            self.window.log(f"灵敏度调整为: {value}%，阈值: {threshold:.6f}")
            print(f"灵敏度调整为: {value}%，阈值: {threshold:.6f}")
    def add_sample(self):
        key = self.window.key_input.text()
        if not key:
            QMessageBox.warning(self.window, "警告", "请输入按下的键")
            return
        if hasattr(self, 'current_features'):
            self.model.add_sample(key, self.current_features)
            self.window.log(f"添加样本: '{key}'")
            self.window.key_input.clear()
            self.update_sample_count()
        else:
            self.window.log("没有可用的样本，请先按下键盘")
    def train_model(self):
        if self.model.train():
            self.window.log("模型训练完成")
            QMessageBox.information(self.window, "成功", "模型训练完成")
        else:
            self.window.log("训练失败，没有足够的样本")
            QMessageBox.warning(self.window, "警告", "训练失败，没有足够的样本")
    def update_sample_count(self):
        counts = self.model.get_sample_count()
        if counts:
            count_str = ", ".join([f"'{k}': {v}" for k, v in counts.items()])
            self.window.log(f"当前样本数: {count_str}")
    def run(self):
        return self.app.exec_()
if __name__ == "__main__":
    app = KeyboardSoundApp()
    sys.exit(app.run())