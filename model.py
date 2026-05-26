"""CRNN model for captcha recognition."""

from __future__ import annotations

import torch
import torch.nn as nn

from charset import NUM_CLASSES


class CRNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, hidden_size: int = 256) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.rnn = nn.LSTM(
            input_size=512 * 3,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=False,
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.cnn(inputs)
        batch_size, channels, height, width = features.size()
        features = features.permute(3, 0, 1, 2)
        features = features.reshape(width, batch_size, channels * height)

        recurrent, _ = self.rnn(features)
        output = self.fc(recurrent)
        log_probs = output.log_softmax(dim=2)
        return log_probs
