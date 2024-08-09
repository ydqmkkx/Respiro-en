import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.models import Conformer
from intervaltree import Interval, IntervalTree

    
def zcr_extractor(wav, win_length, hop_length):
    pad_length = win_length // 2
    wav = np.pad(wav, (pad_length, pad_length), 'constant')
    num_frames = 1 + (wav.shape[0] - win_length) // hop_length
    zcrs = np.zeros(num_frames)
    for i in range(num_frames):
        start = i * hop_length
        end = start + win_length
        zcr = np.abs(wav[start+1:end]-wav[start:end-1])
        zcr = np.sum(zcr) * 0.5 / win_length
        zcrs[i] = zcr
    return zcrs.astype(np.float32)

def feature_extractor(wav, sr=16000):
    mel = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=int(sr*0.025), hop_length=int(sr*0.01), n_mels=128)
    mel = librosa.power_to_db(mel, ref=np.max)
    zcr = zcr_extractor(wav, win_length=int(sr*0.025), hop_length=int(sr*0.01))
    vms = np.var(mel, axis=0)

    mel = torch.tensor(mel).unsqueeze(0)
    zcr = torch.tensor(zcr).unsqueeze(0)
    vms = torch.tensor(vms).unsqueeze(0)

    zcr = zcr.unsqueeze(1).expand(-1, 128, -1)
    vms = torch.var(mel, dim=1).unsqueeze(1).expand(-1, mel.shape[1], -1)

    feature = torch.stack((mel, vms, zcr), dim=1)
    length = torch.tensor([zcr.shape[-1]])
    return feature, length


class Conv2dDownsampling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(Conv2dDownsampling, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        
    def forward(self, x, length):
        keep_dim_padding = 1 - x.shape[-1] % 2 # odd: 0; even: 1
        x = F.pad(x, (0, keep_dim_padding, 0, 0))
        x = self.conv1(x)
        length = (length - 3 + keep_dim_padding) // 2 + 1
        
        keep_dim_padding = 1 - x.shape[-1] % 2
        x = F.pad(x, (0, keep_dim_padding, 0, 0))
        x = self.conv2(x)
        length = (length - 3 + keep_dim_padding) // 2 + 1
        return x, length

class Conv1dUpsampling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(Conv1dUpsampling, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(out_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.deconv(x)
        return x

class DetectionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.downsampling = Conv2dDownsampling(3, 1)
        self.upsampling = Conv1dUpsampling(128, 128)
        self.linear = nn.Linear(31, 128)
        self.dropout = nn.Dropout(0.1)
        
        # Conformer
        self.conformer = Conformer(input_dim=128, num_heads=4, ffn_dim=256, num_layers=8, depthwise_conv_kernel_size=31, dropout=0.1)
        
        # BiLSTM
        self.lstm = nn.LSTM(input_size=128, hidden_size=256, bidirectional=True, batch_first=True)
        
        self.fc = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, length):
        sequence = x.shape[-1]
        x, length = self.downsampling(x, length) # x.shape: [B, 1, 31, S]
        x = x.squeeze(1).transpose(1, 2).contiguous() # x.shape: [B, S, 31]
        x = self.linear(x) # x.shape: [B, S, 128]
        x = self.dropout(x)
        x = self.conformer(x, length)[0]
        x = x.transpose(1, 2).contiguous()
        x = self.upsampling(x)
        x = x.transpose(1, 2).contiguous()
        x = self.lstm(x)[0]
        x = self.fc(x)
        x = self.sigmoid(x.squeeze(-1))
        return x[:, :sequence]


class BreathDetector:
    def __init__(self, model, device=None):
        super().__init__()
        self.model = model
        if not device:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def __call__(self, wav_path, threshold=0.064, min_length=20):
        wav, sr = librosa.load(wav_path, sr=16000)
        feature, length = feature_extractor(wav, sr)
        feature, length = feature.to(self.device), length.to(self.device)
        output = self.model(feature, length)

        prediction = (output[0] > threshold).nonzero().squeeze().tolist()
        tree = IntervalTree()
        if isinstance(prediction, list) and len(prediction)>1:
            diffs = np.diff(prediction)
            splits = np.where(diffs != 1)[0] + 1
            splits = np.split(prediction, splits)
            splits = list(filter(lambda split: len(split)>min_length, splits))
            for split in splits:
                if split[-1]*0.01>split[0]*0.01:
                    tree.add(Interval(round(split[0]*0.01, 2), round(split[-1]*0.01, 2)))
        return tree