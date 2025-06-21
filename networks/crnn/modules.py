import math

import torch
import torch.nn as nn

from my_utils.data_preprocessing import NUM_CHANNELS, IMG_HEIGHT


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        layers = [
            # First block
            nn.Conv2d(NUM_CHANNELS, 8, (10, 2), padding="same", bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 2)),
            # Second block
            nn.Conv2d(8, 8, (8, 5), padding="same", bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d((2, 1)),
        ]
        self.backbone = nn.Sequential(*layers)
        self.width_reduction = 2
        self.height_reduction = 2**2
        self.out_channels = 8

    def forward(self, x):
        return self.backbone(x)


class RNN(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(RNN, self).__init__()
        self.blstm = nn.LSTM(
            input_size,
            256,
            num_layers=2,
            batch_first=True,
            dropout=0.5,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(256 * 2, output_size)

    def forward(self, x):
        x, _ = self.blstm(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x


class CRNN(nn.Module):
    def __init__(self, output_size: int, frame_multiplier_factor: int):
        super(CRNN, self).__init__()
        # CNN
        self.cnn = CNN()
        # RNN
        self.num_frame_repeats = self.cnn.width_reduction * frame_multiplier_factor
        self.rnn_input_size = self.cnn.out_channels * (IMG_HEIGHT // self.cnn.height_reduction)
        self.rnn = RNN(input_size=self.rnn_input_size, output_size=output_size)

    def forward(self, x):
        # CNN
        # x: [b, NUM_CHANNELS, IMG_HEIGHT, w]
        x = self.cnn(x)
        # x: [b, self.cnn.out_channels, nh = IMG_HEIGHT // self.height_reduction, nw = w // self.width_reduction]
        # Prepare for RNN
        b, _, _, w = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.reshape(b, w, self.rnn_input_size)
        # x: [b, nw, self.cnn.out_channels * nh]
        x = torch.repeat_interleave(x, repeats=self.num_frame_repeats, dim=1)
        # x: [b, self.num_frame_repeats * nw, self.cnn.out_channels * nh]
        # RNN
        x = self.rnn(x)
        # x: [b, self.num_frame_repeats * nw, output_size]
        return x
