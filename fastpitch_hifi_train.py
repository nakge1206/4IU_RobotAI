# fastpitch_hifi_train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import weight_norm
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import json

# Configurations
SAMPLING_RATE = 16000
BATCH_SIZE = 8
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_DIR = 'checkpoints'
RESUME_PATH = None  # 이어서 학습하려면 체크포인트 경로 입력

# 1.Utils
class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def simple_tokenizer(text):
    return torch.tensor([ord(c) % 256 for c in text], dtype=torch.long)

# 2.Dataset
class TTSDatasetFromFolder(Dataset):
    def __init__(self, folder_path):
        self.samples = []
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLING_RATE,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_mels=80
        )
        for filename in os.listdir(folder_path):
            if filename.endswith('.json'):
                json_path = os.path.join(folder_path, filename)
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    wav_filename = data['File']['FileName']
                    transcription = data['Transcription']['LabelText']
                    wav_path = os.path.join(folder_path, wav_filename)
                    if os.path.exists(wav_path):
                        self.samples.append((wav_path, transcription))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        wav_path, text = self.samples[idx]
        wav, sr = torchaudio.load(wav_path)
        if sr != SAMPLING_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLING_RATE)
        mel = self.mel_transform(wav).squeeze(0)
        return mel, text

def collate_fn(batch):
    mels, texts = zip(*batch)
    return list(mels), list(texts)

# 3.Model

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
        return mel, waveform

# 4.Trainer
def train():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    writer = SummaryWriter(log_dir='runs')

    train_dataset = TTSDatasetFromFolder('dataset/train')
    valid_dataset = TTSDatasetFromFolder('dataset/valid')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = JointFastPitchHiFiGAN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    mel_loss_fn = nn.MSELoss()

    early_stopping = EarlyStopping(patience=5)
    start_epoch = 1

    if RESUME_PATH:
        checkpoint = torch.load(RESUME_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")

    best_val_loss = float('inf')

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0

        for mels, texts in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
            text_tokens = [simple_tokenizer(t) for t in texts]
            text_tokens = nn.utils.rnn.pad_sequence(text_tokens, batch_first=True).to(DEVICE)
            mels = torch.stack(mels).to(DEVICE)

            if text_tokens.size(1) != mels.size(2):
                min_len = min(text_tokens.size(1), mels.size(2))
                text_tokens = text_tokens[:, :min_len]
                mels = mels[:, :, :min_len]

            pred_mel, _ = model(text_tokens)
            loss = mel_loss_fn(pred_mel, mels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for mels, texts in valid_loader:
                text_tokens = [simple_tokenizer(t) for t in texts]
                text_tokens = nn.utils.rnn.pad_sequence(text_tokens, batch_first=True).to(DEVICE)
                mels = torch.stack(mels).to(DEVICE)

                if text_tokens.size(1) != mels.size(2):
                    min_len = min(text_tokens.size(1), mels.size(2))
                    text_tokens = text_tokens[:, :min_len]
                    mels = mels[:, :, :min_len]

                pred_mel, _ = model(text_tokens)
                loss = mel_loss_fn(pred_mel, mels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(valid_loader)

        print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        writer.add_scalars('Loss', {'Train': avg_train_loss, 'Validation': avg_val_loss}, epoch)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss
            }, os.path.join(CHECKPOINT_DIR, 'best_model.pth'))
            print("Saved best model!")

        # Early stopping
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

# 5.Main 실행
if __name__ == "__main__":
    train()
