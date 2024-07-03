import torch
from tacotron2.model import Tacotron2
from tacotron2.hparams import hparams
from text import text_to_sequence
import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf

def text_to_mel(text, model, hparams):
    """将文本转换为Mel频谱图"""
    # 文本预处理
    sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()

    # 生成Mel频谱图
    with torch.no_grad():
        mel_outputs, mel_outputs_postnet, _, _ = model.inference(sequence)
        mel = mel_outputs_postnet.float().data.cpu().numpy()[0]

    return mel

def mel_to_audio(mel_spectrogram, sr=22050, n_fft=2048, win_length=2048, hop_length=512, volume_gain=100):
    # 从Mel频谱图到STFT幅度
    mel_inv = librosa.feature.inverse.mel_to_stft(mel_spectrogram, sr=sr, n_fft=n_fft)
    
    # 应用Griffin-Lim算法重建音频
    audio = librosa.griffinlim(mel_inv, n_iter=32, hop_length=hop_length, win_length=win_length)
    
    # 增大音量
    audio *= volume_gain
    
    # 限幅处理，防止超出[-1.0, 1.0]范围
    audio = np.clip(audio, -1.0, 1.0)
    
    return audio

# 加载模型
model = Tacotron2(hparams)
# model.load_state_dict(torch.load(r"tacotron2_statedict.pt")["state_dict"])
model.load_state_dict(torch.load(r"output_pth\checkpoint_33000.pth")["state_dict"])
model.cuda().eval()


if __name__ == '__main__':
    # 输入文本
    text = "So cry no more oh my beloved"

    # 文本到Mel频谱图
    mel = text_to_mel(text, model, hparams)

    # 从Mel频谱图重建音频
    audio = mel_to_audio(mel)

    # 保存重建的音频
    sf.write(r'output_audio\reconstructed_audio.wav', audio, 22050)

    # 显示Mel频谱图
    plt.figure(figsize=(10, 4))
    plt.imshow(mel, aspect='auto', origin='lower', cmap='viridis')
    plt.title('Mel Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()