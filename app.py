from flask import Flask, request, render_template, send_from_directory
import torch
from tacotron2.model import Tacotron2
from tacotron2.hparams import hparams
from text import text_to_sequence
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import io
import os
from datetime import datetime

app = Flask(__name__)

def text_to_mel(text, model, hparams):
    sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()

    with torch.no_grad():
        mel_outputs, mel_outputs_postnet, _, _ = model.inference(sequence)
        mel = mel_outputs_postnet.float().data.cpu().numpy()[0]

    return mel

def mel_to_audio(mel_spectrogram, sr=22050, n_fft=2048, win_length=2048, hop_length=512, volume_gain=100):
    mel_inv = librosa.feature.inverse.mel_to_stft(mel_spectrogram, sr=sr, n_fft=n_fft)
    audio = librosa.griffinlim(mel_inv, n_iter=32, hop_length=hop_length, win_length=win_length)
    audio *= volume_gain
    audio = np.clip(audio, -1.0, 1.0)
    return audio

def save_mel_spectrogram(mel, path):
    plt.figure(figsize=(10, 4))
    plt.imshow(mel, aspect='auto', origin='lower', cmap='viridis')
    plt.title('Mel Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

# 加载模型
model = Tacotron2(hparams)
# model.load_state_dict(torch.load(r"tacotron2_statedict.pt")["state_dict"])
model.load_state_dict(torch.load(r"output_pth\checkpoint_33000.pth")["state_dict"])
model.cuda().eval()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    text = request.form['text']
    mel = text_to_mel(text, model, hparams)
    audio = mel_to_audio(mel)
    
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    audio_path = os.path.join("results/audio", f"audio_{current_time}.wav")
    mel_path = os.path.join("results/mel", f"mel_{current_time}.png")

    os.makedirs(os.path.dirname(audio_path), exist_ok=True)
    os.makedirs(os.path.dirname(mel_path), exist_ok=True)

    sf.write(audio_path, audio, 22050)
    save_mel_spectrogram(mel, mel_path)
    
    return {
        "audio_path": f"/results/audio/audio_{current_time}.wav",
        "mel_path": f"/results/mel/mel_{current_time}.png"
    }

@app.route('/results/audio/<filename>')
def get_audio(filename):
    return send_from_directory('results/audio', filename)

@app.route('/results/mel/<filename>')
def get_mel(filename):
    return send_from_directory('results/mel', filename)

if __name__ == '__main__':
    app.run(debug=True)