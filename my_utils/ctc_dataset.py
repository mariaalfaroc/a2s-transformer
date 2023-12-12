import json
import os

import torch
import lightning as L
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from my_utils.encoding_convertions import krnParser
from my_utils.data_preprocessing import preprocess_audio, ctc_batch_preparation

DATASETS = ["Quartets", "Beethoven", "Mozart", "Haydn"]


class CTCDataModule(L.LightningDataModule):
    def __init__(
        self,
        ds_name: str,
        batch_size: int = 16,
        num_workers: int = 20,
        width_reduction: int = None,
    ):
        super(CTCDataModule).__init__()
        self.ds_name = ds_name
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Must be overrided when the model is created
        self.width_reduction = width_reduction

    def setup(self, stage: str):
        assert (
            self.width_reduction is not None
        ), "width_reduction is None. Override the width_reduction attribute with that of the model."

        if stage == "fit":
            self.train_ds = CTCDataset(
                ds_name=self.ds_name,
                partition_type="train",
                width_reduction=self.width_reduction,
            )
            self.val_ds = CTCDataset(
                ds_name=self.ds_name,
                partition_type="val",
                width_reduction=self.width_reduction,
            )

        if stage == "test" or stage == "predict":
            self.test_ds = CTCDataset(
                ds_name=self.ds_name,
                partition_type="test",
                width_reduction=self.width_reduction,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=ctc_batch_preparation,
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


####################################################################################################


class CTCDataset(Dataset):
    def __init__(
        self,
        ds_name: str,
        partition_type: str,
        width_reduction: int = 2,
        use_voice_change_token: bool = False,
    ):
        self.ds_name = ds_name.lower()
        self.partition_type = partition_type
        self.width_reduction = width_reduction
        self.use_voice_change_token = use_voice_change_token

        # Initialize krn parser
        self.krn_parser = krnParser()

        # Check dataset name
        assert self.ds_name in DATASETS, f"Invalid dataset name: {self.ds_name}"

        # Check partition type
        assert self.partition_type in [
            "train",
            "val",
            "test",
        ], f"Invalid partition type: {self.partition_type}"

        # Get audios and transcripts files
        self.X, self.Y = self.get_audios_and_transcripts_files()

        # Check and retrieve vocabulary
        vocab_folder = os.path.join("Quartets", "vocabs")
        os.makedirs(vocab_folder, exist_ok=True)
        vocab_name = self.ds_name + "_w2i"
        vocab_name += "_withvc" if self.use_voice_change_token else ""
        vocab_name += ".json"
        self.w2i_path = os.path.join(vocab_folder, vocab_name)
        self.w2i, self.i2w = self.check_and_retrieve_vocabulary()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # CTC training setting
        x = preprocess_audio(path=self.X[idx])
        y = self.preprocess_ctc_transcript(path=self.Y[idx])
        if self.partition_type == "train":
            # x.shape = [channels, height, width]
            return x, x.shape[2] // self.width_reduction, y, len(y)
        return x, y

    def preprocess_ctc_transcript(self, path: str):
        y = self.krn_parser.convert(src_file=path)
        if not self.use_voice_change_token:
            y = [w for w in y if w != self.krn_parser.voice_change]
        y = [self.w2i[w] for w in y]
        return torch.tensor(y, dtype=torch.int32)

    def get_audios_and_transcripts_files(self):
        partition_file = f"Quartets/partitions/{self.ds_name}/{self.partition_type}.txt"

        audios = []
        transcripts = []
        with open(partition_file, "r") as file:
            for s in file.read().splitlines():
                s = s.strip()
                audios.append(f"Quartets/flac/{s}.flac")
                transcripts.append(f"Quartets/krn/{s}.krn")
        return audios, transcripts

    def check_and_retrieve_vocabulary(self):
        w2i = {}
        i2w = {}

        if os.path.isfile(self.w2i_path):
            with open(self.w2i_path, "r") as file:
                w2i = json.load(file)
            i2w = {v: k for k, v in w2i.items()}
        else:
            w2i, i2w = self.make_vocabulary()
            with open(self.w2i_path, "w") as file:
                json.dump(w2i, file)

        return w2i, i2w

    def make_vocabulary(self):
        vocab = []
        for partition_type in ["train", "val", "test"]:
            partition_file = f"Quartets/partitions/{self.ds_name}/{partition_type}.txt"
            with open(partition_file, "r") as file:
                for s in file.read().splitlines():
                    s = s.strip()
                    vocab.extend(
                        self.krn_parser.convert(src_file=f"Quartets/krn/{s}.krn")
                    )
        vocab = sorted(set(vocab))
        if not self.use_voice_change_token:
            del vocab[vocab.index(self.krn_parser.voice_change)]

        w2i = {}
        i2w = {}
        for i, w in enumerate(vocab):
            w2i[w] = i + 1
            i2w[i + 1] = w
        w2i["<PAD>"] = 0
        i2w[0] = "<PAD>"

        return w2i, i2w
