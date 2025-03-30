# Hear Your Key - 键盘声音识别

![键盘声音识别](https://img.shields.io/badge/键盘声音识别-v1.0-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![PyQt5](https://img.shields.io/badge/PyQt5-5.15+-orange)
![Librosa](https://img.shields.io/badge/Librosa-0.9+-yellow)

Hear Your Key 能够通过麦克风捕获键盘按键声音，并识别出按下的是哪个按键。该工具采用PyQt5构建用户界面，使用librosa进行音频特征提取，通过随机森林算法实现按键声音的分类识别。

## 安装

### 环境要求

- Python 3+

### 安装步骤

1. 克隆仓库到本地

```bash
git clone https://github.com/yourusername/hear-your-key.git
cd hear-your-key
```

2. 安装依赖包
```bash
pip install -r requirements.txt
```

3. 运行程序
```bash
python main.py
```

## 使用说明
### 学习模式
1. 选择"学习模式"
2. 调整麦克风和灵敏度设置
3. 按下键盘按键，当程序检测到按键声音时，在输入框中输入对应的按键名称
4. 点击"添加样本"保存该按键的声音特征
5. 对每个需要识别的按键重复上述步骤，建议每个按键至少添加5-10个样本
6. 点击"训练模型"开始训练识别模型

### 匹配模式
1. 完成模型训练后，切换到"匹配模式"
2. 按下键盘按键，程序将自动识别并显示按键名称及置信度
### 灵敏度调节
- 如果程序无法检测到按键声音，请增加灵敏度（向右调整滑块）
- 如果环境噪音导致误触发，请降低灵敏度（向左调整滑块）
## 项目结构
- main.py - 主程序入口
- ui.py - 用户界面实现
- audio_recorder.py - 音频录制和按键检测
- feature_extractor.py - 音频特征提取
- model.py - 机器学习模型实现
- data_manager.py - 数据管理和持久化
