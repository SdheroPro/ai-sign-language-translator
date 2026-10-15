# 🤟 Ars Signorum — Sign Language Interpreter

[![Python Version](https://img.shields.io/badge/python-3.9--3.13-blue.svg)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.14-green.svg)](https://google.github.io/mediapipe/)
[![CustomTkinter](https://img.shields.io/badge/UI-CustomTkinter-orange.svg)](https://github.com/TomSchimansky/CustomTkinter)
[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)

**Ars Signorum** is a real-time computer vision and machine learning application that interprets sign language through a webcam, translating hand gestures, ASL alphabet letters, and custom-trained holds/motions into text on screen.

It integrates **MediaPipe Tasks**, a pretrained **ASL A-Z TFLite keypoint classifier**, a **k-NN custom gesture store**, **stabilizer filtering**, and a polished **CustomTkinter interface**.

---

## ✨ Key Recognition Modes

* **Gestures:** Pretrained Google Gesture Recognizer handling 8 built-in gestures (Open Palm, Thumbs Up/Down, Closed Fist, Pointing Up, Victory, I-Love-You) mapped to customizable words.
* **Alphabet:** Pretrained ASL A-Z keypoint classifier (MIT, Muhib-Mehdi). Hold a letter steady to type it, or move your hand out of view to insert a space.
* **Custom Hold:** Teach the app your own static hand shapes.
* **Custom Motion:** Record and recognize dynamic moving gestures using a sliding window and resampling pipeline.

---

## 🛠️ Quick Start & Installation

### Requirements
* **Python:** 3.9 or newer *(recommended)*[cite: 3]
* **Hardware:** Any standard webcam[cite: 3]
* **OS:** Windows, macOS, or Linux *(Note for Linux: requires `python3-tk`)[cite: 3]*

### Running with the Setup Script
Clone this repository and run `sign_language_setup.py`. The automated setup script creates a private virtual environment (`.venv`), installs all required packages (`mediapipe`, `opencv-python`, `customtkinter`, `numpy`, `pillow`), unpacks the bundled ASL TFLite model, and launches the application[cite: 3]:

```bash
# Clone repository
git clone https://github.com/SdheroPro/ai-sign-language-translator.git
cd ai-sign-language-translator

# Run automated setup & launch app
python sign_language_setup.py
