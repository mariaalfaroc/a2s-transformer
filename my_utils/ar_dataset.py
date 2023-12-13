import torch
import lightning as L
from torch.utils.data import DataLoader

from my_utils.ctc_dataset import CTCDataset
from my_utils.data_preprocessing import preprocess_audio, ar_batch_preparation

SOS_TOKEN = "<SOS>"  # Start-of-sequence token
EOS_TOKEN = "<EOS>"  # End-of-sequence token


class ARDataModule(L.LightningDataModule):
    def __init__(
        self,
        ds_name: str,
        use_voice_change_token: bool = False,
        batch_size: int = 16,
        num_workers: int = 20,
    ):
        super(ARDataModule).__init__()
        self.ds_name = ds_name
        self.use_voice_change_token = use_voice_change_token
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str):
        if stage == "fit":
            self.train_ds = ARDataset(
                ds_name=self.ds_name,
                partition_type="train",
                use_voice_change_token=self.use_voice_change_token,
            )
            self.val_ds = ARDataset(
                ds_name=self.ds_name,
                partition_type="val",
                use_voice_change_token=self.use_voice_change_token,
            )

        if stage == "test" or stage == "predict":
            self.test_ds = ARDataset(
                ds_name=self.ds_name,
                partition_type="test",
                use_voice_change_token=self.use_voice_change_token,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=ar_batch_preparation,
        )  # prefetch_factor=2

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
        )  # prefetch_factor=2

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
        )  # prefetch_factor=2

    def predict_dataloader(self):
        print("Using test_dataloader for predictions.")
        return self.test_dataloader(self)

    def get_w2i_and_i2w(self):
        return self.train_ds.w2i, self.train_ds.i2w

    def get_max_seq_len(self):
        return self.train_ds.max_seq_len

    def get_max_audio_len(self):
        return self.train_ds.max_audio_len


####################################################################################################


class ARDataset(CTCDataset):
    def __init__(
        self,
        ds_name: str,
        partition_type: str,
        use_voice_change_token: bool = False,
    ):
        self.ds_name = ds_name.lower()
        self.partition_type = partition_type
        self.use_voice_change_token = use_voice_change_token
        self.setup(vocab_name="ar_w2i")

    def __getitem__(self, idx):
        x = preprocess_audio(path=self.X[idx])
        y = self.preprocess_transcript(path=self.Y[idx])
        return x, y

    def preprocess_transcript(self, path: str):
        y = self.krn_parser.convert(src_file=path)
        y = [SOS_TOKEN] + y + [EOS_TOKEN]
        if not self.use_voice_change_token:
            y = [w for w in y if w != self.krn_parser.voice_change]
        y = [self.w2i[w] for w in y]
        return torch.tensor(y, dtype=torch.int64)

    def make_vocabulary(self):
        vocab = self.get_unique_tokens_and_max_seq_len()[0]
        vocab = [SOS_TOKEN, EOS_TOKEN] + vocab
        vocab = sorted(vocab)

        w2i = {}
        i2w = {}
        for i, w in enumerate(vocab):
            w2i[w] = i + 1
            i2w[i + 1] = w
        w2i["<PAD>"] = 0
        i2w[0] = "<PAD>"

        return w2i, i2w
