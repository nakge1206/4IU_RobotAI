# fastpitch.py

import torch
import torchaudio
import os
import torch.nn as nn
from torch.nn.utils import weight_norm
from tqdm import tqdm

# Config
SAMPLING_RATE = 16000
MODEL_PATH = os.path.join(os.getcwd(), 'checkpoints', 'best_model.pth')
OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Utils
def simple_tokenizer(text):
    return torch.tensor([ord(c) % 256 for c in text], dtype=torch.long)

# Model
class ResBlock1(nn.Module):
    def __init__(self, channels, kernel_size, dilations):
        super().__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, padding=dilation*(kernel_size-1)//2, dilation=dilation))
            for dilation in dilations
        ])
        self.convs2 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, padding=1*(kernel_size-1)//2, dilation=1))
            for _ in dilations
        ])
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        for conv1, conv2 in zip(self.convs1, self.convs2):
            xt = self.activation(x)
            xt = conv1(xt)
            xt = self.activation(xt)
            xt = conv2(xt)
            x = xt + x
        return x

class JointFastPitchHiFiGAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Embedding(256, 256)
        self.duration_predictor = nn.Linear(256, 1)
        self.pitch_predictor = nn.Linear(256, 1)
        self.energy_predictor = nn.Linear(256, 1)
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 80)
        )

        self.num_kernels = 3
        self.num_upsamples = 4

        self.conv_pre = weight_norm(nn.Conv1d(80, 512, kernel_size=7, stride=1, padding=3))
        self.ups = nn.ModuleList([
            weight_norm(nn.ConvTranspose1d(512, 256, 16, 8, padding=4)),
            weight_norm(nn.ConvTranspose1d(256, 128, 16, 8, padding=4)),
            weight_norm(nn.ConvTranspose1d(128, 64, 4, 2, padding=1)),
            weight_norm(nn.ConvTranspose1d(64, 32, 4, 2, padding=1)),
        ])
        self.resblocks = nn.ModuleList([
            ResBlock1(ch, k, [1, 3, 5])
            for ch in [256, 256, 256, 128, 128, 128, 64, 64, 64, 32, 32, 32]
            for k in [3, 7, 11]
        ])
        self.conv_post = weight_norm(nn.Conv1d(32, 1, kernel_size=7, stride=1, padding=3))
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, text_tokens):
        x = self.encoder(text_tokens)
        _ = self.duration_predictor(x)
        _ = self.pitch_predictor(x)
        _ = self.energy_predictor(x)
        mel = self.decoder(x)
        mel = mel.transpose(1, 2)

        x = self.conv_pre(mel)
        for i in range(self.num_upsamples):
            x = self.activation(x)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                idx = i * self.num_kernels + j
                xs = self.resblocks[idx](x) if xs is None else xs + self.resblocks[idx](x)
            x = xs / self.num_kernels
        x = self.activation(x)
        x = self.conv_post(x)
        waveform = torch.tanh(x)
        return waveform

# Synthesizer
def synthesize(text, output_wav_name='output.wav'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = JointFastPitchHiFiGAN().to(device)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Checkpoint not found at {MODEL_PATH}. Please train the model first.")

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    tokens = simple_tokenizer(text).unsqueeze(0).to(device)

    with torch.no_grad():
        print(f"Generating speech for: \"{text}\"")
        waveform = model(tokens)
        waveform = waveform.squeeze(0).cpu()

    output_path = os.path.join(OUTPUT_DIR, output_wav_name)
    torchaudio.save(output_path, waveform, sample_rate=SAMPLING_RATE)
    print(f"Saved synthesized audio to {output_path}")

# Main
if __name__ == "__main__":
    sample_texts = [
        "오늘은 정말 좋은 날씨입니다.",
        "안녕하세요. 만나서 반갑습니다.",
        "FastPitch와 HiFiGAN을 합친 모델입니다."
    ]

    for idx, text in enumerate(tqdm(sample_texts, desc="Synthesizing")):
        synthesize(text, output_wav_name=f'sample_{idx+1}.wav')
