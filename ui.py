from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QSlider, QRadioButton, QPushButton, QLineEdit, QTextEdit, QGroupBox
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
class AudioVisualizer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.audio_buffer = np.zeros(8192)
        self.update_counter = 0
        self.update_rate = 3
    def setup_ui(self):
        layout = QVBoxLayout()
        plt.close('all')
        self.waveform_fig = plt.figure(figsize=(5, 2))
        self.waveform_canvas = FigureCanvas(self.waveform_fig)
        self.waveform_ax = self.waveform_fig.add_subplot(111)
        self.waveform_ax.set_title("波形图")
        self.waveform_ax.set_ylim(-1, 1)
        self.waveform_ax.set_xlim(0, 8192)
        self.waveform_ax.set_facecolor('#f0f0f0')
        self.waveform_ax.grid(True, alpha=0.3)
        self.waveform_line, = self.waveform_ax.plot(np.arange(8192), np.zeros(8192), '-', lw=0.8, color='#3366cc')
        self.waveform_fill = self.waveform_ax.fill_between(np.arange(8192), np.zeros(8192), -np.zeros(8192), alpha=0.4, color='#3366cc')
        self.waveform_fig.tight_layout()
        self.spec_fig = plt.figure(figsize=(5, 3))
        self.spec_canvas = FigureCanvas(self.spec_fig)
        self.spec_ax = self.spec_fig.add_subplot(111)
        self.spec_ax.set_title("频谱图")
        self.spec_img = self.spec_ax.imshow(np.zeros((128, 128)), aspect='auto', origin='lower', cmap='viridis')
        self.spec_fig.tight_layout()
        layout.addWidget(self.waveform_canvas)
        layout.addWidget(self.spec_canvas)
        self.setLayout(layout)
    def update_waveform(self, audio_data):
        try:
            self.update_counter += 1
            if self.update_counter % self.update_rate != 0:
                audio_len = len(audio_data)
                if audio_len < 8192:
                    chunk_size = audio_len // self.update_rate
                    if chunk_size > 0:
                        self.audio_buffer = np.roll(self.audio_buffer, -chunk_size)
                        self.audio_buffer[-chunk_size:] = audio_data[:chunk_size]
                return
            self.update_counter = 0
            audio_len = len(audio_data)
            if audio_len >= 8192:
                self.audio_buffer = audio_data[-8192:]
            else:
                move_size = min(audio_len, 1024)
                self.audio_buffer = np.roll(self.audio_buffer, -move_size)
                self.audio_buffer[-move_size:] = audio_data[:move_size]
            self.waveform_line.set_ydata(self.audio_buffer)
            for coll in self.waveform_ax.collections[:]:
                coll.remove()
            self.waveform_fill = self.waveform_ax.fill_between(
                np.arange(8192), 
                self.audio_buffer, 
                -0.01,
                where=(self.audio_buffer > -0.01),
                alpha=0.4, 
                color='#3366cc'
            )
            self.waveform_canvas.draw()
        except Exception as e:
            print(f"波形图更新错误: {e}")
    def update_spectrogram(self, spec_data):
        try:
            self.spec_img.set_data(spec_data)
            self.spec_img.set_clim(np.min(spec_data), np.max(spec_data))
            self.spec_canvas.draw()
        except Exception as e:
            print(f"频谱图更新错误: {e}")
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("键盘声音识别")
        self.setGeometry(100, 100, 800, 600)
        self.setup_ui()
    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        top_layout = QHBoxLayout()
        control_layout = QVBoxLayout()
        mode_group = QGroupBox("模式")
        mode_layout = QVBoxLayout(mode_group)
        self.learn_mode_radio = QRadioButton("学习模式")
        self.match_mode_radio = QRadioButton("匹配模式")
        self.learn_mode_radio.setChecked(True)
        mode_layout.addWidget(self.learn_mode_radio)
        mode_layout.addWidget(self.match_mode_radio)
        control_layout.addWidget(mode_group)
        mic_group = QGroupBox("麦克风")
        mic_layout = QVBoxLayout(mic_group)
        self.mic_combo = QComboBox()
        mic_layout.addWidget(self.mic_combo)
        control_layout.addWidget(mic_group)
        sensitivity_group = QGroupBox("灵敏度")
        sensitivity_layout = QVBoxLayout(sensitivity_group)
        self.sensitivity_slider = QSlider(Qt.Horizontal)
        self.sensitivity_slider.setMinimum(0)
        self.sensitivity_slider.setMaximum(100)
        self.sensitivity_slider.setValue(50)
        sensitivity_layout.addWidget(self.sensitivity_slider)
        control_layout.addWidget(sensitivity_group)
        learn_group = QGroupBox("学习控制")
        learn_layout = QVBoxLayout(learn_group)
        key_layout = QHBoxLayout()
        key_layout.addWidget(QLabel("按键:"))
        self.key_input = QLineEdit()
        key_layout.addWidget(self.key_input)
        learn_layout.addLayout(key_layout)
        self.add_sample_btn = QPushButton("添加样本")
        self.train_model_btn = QPushButton("训练模型")
        learn_layout.addWidget(self.add_sample_btn)
        learn_layout.addWidget(self.train_model_btn)
        control_layout.addWidget(learn_group)
        result_group = QGroupBox("识别结果")
        result_layout = QVBoxLayout(result_group)
        self.result_label = QLabel("等待识别...")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        result_layout.addWidget(self.result_label)
        control_layout.addWidget(result_group)
        control_layout.addStretch()
        top_layout.addLayout(control_layout, 1)
        self.visualizer = AudioVisualizer()
        top_layout.addWidget(self.visualizer, 3)
        main_layout.addLayout(top_layout)
        log_group = QGroupBox("日志")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        main_layout.addWidget(log_group)
    def log(self, message):
        self.log_text.append(message)
    def update_result(self, key, confidence=None):
        if confidence:
            self.result_label.setText(f"{key} ({confidence:.2f})")
        else:
            self.result_label.setText(key)