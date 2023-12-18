import random

import torch
from torch.nn import CTCLoss
from torchinfo import summary
from lightning.pytorch import LightningModule

from networks.crnn.modules import CRNN
from my_utils.metrics import compute_metrics
from my_utils.data_preprocessing import IMG_HEIGHT, NUM_CHANNELS


class CTCTrainedCRNN(LightningModule):
    def __init__(self, w2i, i2w, ytest_i2w=None, num_frame_repeats=16):
        super(CTCTrainedCRNN, self).__init__()
        # Save hyperparameters
        self.save_hyperparameters()
        # Dictionaries
        self.w2i = w2i
        self.i2w = i2w
        self.ytest_i2w = ytest_i2w if ytest_i2w is not None else i2w
        # Model
        self.model = CRNN(
            output_size=len(self.w2i) + 1, num_frame_repeats=num_frame_repeats
        )
        self.width_reduction = self.model.cnn.width_reduction
        self.summary()
        # Loss: the target index cannot be blank!
        self.compute_ctc_loss = CTCLoss(blank=len(self.w2i), zero_infinity=False)
        # Predictions
        self.Y = []
        self.YHat = []

    def summary(self):
        summary(self.model, input_size=[1, NUM_CHANNELS, IMG_HEIGHT, 256])

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-6)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, xl, y, yl = batch
        yhat = self.model(x)
        # ------ CTC Requirements ------
        # yhat: [batch, frames, vocab_size]
        yhat = yhat.log_softmax(dim=2)
        yhat = yhat.permute(1, 0, 2).contiguous()
        # ------------------------------
        loss = self.compute_ctc_loss(yhat, y, xl, yl)
        self.log("train_loss", loss, prog_bar=True, logger=True, on_epoch=True)
        return loss

    def ctc_greedy_decoder(self, y_pred, i2w):
        # y_pred = [seq_len, num_classes]
        # Best path
        y_pred_decoded = torch.argmax(y_pred, dim=1)
        # Merge repeated elements
        y_pred_decoded = torch.unique_consecutive(y_pred_decoded, dim=0).tolist()
        # Convert to string; len(i2w) -> CTC-blank
        y_pred_decoded = [i2w[i] for i in y_pred_decoded if i != len(i2w)]
        return y_pred_decoded

    def validation_step(self, batch, batch_idx):
        x, y = batch  # batch_size = 1
        # Model prediction (decoded using the vocabulary on which it was trained)
        yhat = self.model(x)[0]
        yhat = yhat.log_softmax(dim=-1).detach().cpu()
        yhat = self.ctc_greedy_decoder(yhat, self.i2w)
        # Decoded ground truth
        y = [self.ytest_i2w[i.item()] for i in y[0]]
        # Append to later compute metrics
        self.Y.append(y)
        self.YHat.append(yhat)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_validation_epoch_end(self, name="val", print_random_samples=False):
        metrics = compute_metrics(y_true=self.Y, y_pred=self.YHat)
        for k, v in metrics.items():
            self.log(f"{name}_{k}", v, prog_bar=True, logger=True, on_epoch=True)
        # Print random samples
        if print_random_samples:
            index = random.randint(0, len(self.Y) - 1)
            print(f"Ground truth - {self.Y[index]}")
            print(f"Prediction - {self.YHat[index]}")
        # Clear predictions
        self.Y.clear()
        self.YHat.clear()
        return metrics

    def on_test_epoch_end(self):
        return self.on_validation_epoch_end(name="test", print_random_samples=True)
