import math

import torch
from torch.utils.data import DataLoader
from lightning.pytorch import LightningDataModule

from my_utils.ctc_dataset import CTCDataset, load_dataset, SPLITS
from my_utils.data_preprocessing import preprocess_audio, ar_batch_preparation
from networks.transformer.encoder import HEIGHT_REDUCTION, WIDTH_REDUCTION

SOS_TOKEN = "<SOS>"  # Start-of-sequence token
EOS_TOKEN = "<EOS>"  # End-of-sequence token


class ARDataModule(LightningDataModule):
    def __init__(
        self,
        ds_name: str,
        use_voice_change_token: bool = False,
        batch_size: int = 16,
        num_workers: int = 20,
    ):
        super(ARDataModule, self).__init__()
        self.ds_name = ds_name
        self.use_voice_change_token = use_voice_change_token
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Datasets
        # To prevent executing setup() twice
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def setup(self, stage: str):
        if stage == "fit":
            if not self.train_ds:
                self.train_ds = ARDataset(
                    ds_name=self.ds_name,
                    partition_type="train",
                    use_voice_change_token=self.use_voice_change_token,
                )
            if not self.val_ds:
                self.val_ds = ARDataset(
                    ds_name=self.ds_name,
                    partition_type="val",
                    use_voice_change_token=self.use_voice_change_token,
                )

        if stage == "test" or stage == "predict":
            if not self.test_ds:
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
        return self.test_dataloader()

    def get_w2i_and_i2w(self):
        try:
            return self.train_ds.w2i, self.train_ds.i2w
        except AttributeError:
            return self.test_ds.w2i, self.test_ds.i2w

    def get_max_seq_len(self):
        try:
            return self.train_ds.max_seq_len
        except AttributeError:
            return self.test_ds.max_seq_len

    def get_max_audio_len(self):
        try:
            return self.train_ds.max_audio_len
        except AttributeError:
            return self.test_ds.max_audio_len


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
        self.init(vocab_name="ar_w2i")
        self.max_seq_len += 1  # Add 1 for EOS_TOKEN

    def __getitem__(self, idx):
        x = preprocess_audio(
            raw_audio=self.ds[idx]["audio"]["array"], sr=self.ds[idx]["audio"]["sampling_rate"], dtype=torch.float32
        )
        y = self.preprocess_transcript(text=self.ds[idx]["transcript"])
        if self.partition_type == "train":
            return x, self.get_number_of_frames(x), y
        return x, y

    def preprocess_transcript(self, text: str):
        y = self.krn_parser.convert(text=text)
        y = [SOS_TOKEN] + y + [EOS_TOKEN]
        y = [self.w2i[w] for w in y]
        return torch.tensor(y, dtype=torch.int64)

    def make_vocabulary(self):
        full_ds = load_dataset(f"PRAIG/{self.ds_name}-quartets")

        vocab = []
        for split in SPLITS:
            for text in full_ds[split]["transcript"]:
                transcript = self.krn_parser.convert(text=text)
                vocab.extend(transcript)
        vocab = [SOS_TOKEN, EOS_TOKEN] + vocab
        vocab = sorted(set(vocab))

        w2i = {}
        i2w = {}
        for i, w in enumerate(vocab):
            w2i[w] = i + 1
            i2w[i + 1] = w
        w2i["<PAD>"] = 0
        i2w[0] = "<PAD>"

        return w2i, i2w

    def get_number_of_frames(self, audio):
        # audio is the output of preprocess_audio
        # audio.shape = [1, freq_bins, time_frames]
        return math.ceil(audio.shape[1] / HEIGHT_REDUCTION) * math.ceil(audio.shape[2] / WIDTH_REDUCTION)
