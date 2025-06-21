import os
import json
import math

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from lightning.pytorch import LightningDataModule

from my_utils.encoding_convertions import krnParser
from my_utils.data_preprocessing import (
    preprocess_audio,
    ctc_batch_preparation,
    set_pad_index,
)

DATASETS = ["quartets", "beethoven", "mozart", "haydn"]
SPLITS = ["train", "val", "test"]


class CTCDataModule(LightningDataModule):
    def __init__(
        self,
        ds_name: str,
        use_voice_change_token: bool = False,
        batch_size: int = 16,
        num_workers: int = 20,
        width_reduction: int = 2,
    ):
        super(CTCDataModule, self).__init__()
        self.ds_name = ds_name
        self.use_voice_change_token = use_voice_change_token
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.width_reduction = width_reduction  # Must be overrided with that of the model!

        # Datasets
        # To prevent executing setup() twice
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def setup(self, stage: str):
        if stage == "fit":
            if not self.train_ds:
                self.train_ds = CTCDataset(
                    ds_name=self.ds_name,
                    partition_type="train",
                    width_reduction=self.width_reduction,
                    use_voice_change_token=self.use_voice_change_token,
                )
            if not self.val_ds:
                self.val_ds = CTCDataset(
                    ds_name=self.ds_name,
                    partition_type="val",
                    width_reduction=self.width_reduction,
                    use_voice_change_token=self.use_voice_change_token,
                )

        if stage == "test" or stage == "predict":
            if not self.test_ds:
                self.test_ds = CTCDataset(
                    ds_name=self.ds_name,
                    partition_type="test",
                    width_reduction=self.width_reduction,
                    use_voice_change_token=self.use_voice_change_token,
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

    def get_frame_multiplier_factor(self):
        try:
            return self.train_ds.frame_multiplier_factor
        except AttributeError:
            return self.test_ds.frame_multiplier_factor


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
        self.init(vocab_name="ctc_w2i")

    def init(self, vocab_name: str = "w2i"):
        # Initialize krn parser
        self.krn_parser = krnParser(use_voice_change_token=self.use_voice_change_token)

        # Check dataset name
        assert self.ds_name in DATASETS, f"Invalid dataset name: {self.ds_name}"

        # Check partition type
        assert self.partition_type in SPLITS, f"Invalid partition type: {self.partition_type}"

        # Get audios and transcripts files
        self.ds = load_dataset(f"PRAIG/{self.ds_name}-quartets", split=self.partition_type)

        # Check and retrieve vocabulary
        vocab_folder = os.path.join("Quartets", "vocabs")
        os.makedirs(vocab_folder, exist_ok=True)
        vocab_name = self.ds_name + f"_{vocab_name}"
        vocab_name += "_withvc" if self.use_voice_change_token else ""
        vocab_name += ".json"
        self.w2i_path = os.path.join(vocab_folder, vocab_name)
        self.w2i, self.i2w = self.check_and_retrieve_vocabulary()
        # Modify the global PAD_INDEX to match w2i["<PAD>"]
        set_pad_index(self.w2i["<PAD>"])

        # Check and retrive max lengths
        # Set max_seq_len, max_audio_len and frame_multiplier_factor
        max_lens_folder = os.path.join("Quartets", "max_lens")
        os.makedirs(max_lens_folder, exist_ok=True)
        max_lens_name = vocab_name
        self.max_lens_path = os.path.join(max_lens_folder, max_lens_name)
        max_lens = self.check_and_retrieve_max_lens()
        self.max_seq_len = max_lens["max_seq_len"]
        self.max_audio_len = max_lens["max_audio_len"]
        self.frame_multiplier_factor = max_lens["max_frame_multiplier_factor"]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x = preprocess_audio(
            raw_audio=self.ds[idx]["audio"]["array"], sr=self.ds[idx]["audio"]["sampling_rate"], dtype=torch.float32
        )
        y = self.preprocess_transcript(text=self.ds[idx]["transcript"])
        if self.partition_type == "train":
            # x.shape = [channels, height, width]
            return (
                x,
                (x.shape[2] // self.width_reduction) * self.width_reduction * self.frame_multiplier_factor,
                y,
                len(y),
            )
        return x, y

    def preprocess_transcript(self, text: str):
        y = self.krn_parser.convert(text=text)
        y = [self.w2i[w] for w in y]
        return torch.tensor(y, dtype=torch.int32)

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
        full_ds = load_dataset(f"PRAIG/{self.ds_name}-quartets")

        vocab = []
        for split in SPLITS:
            for text in full_ds[split]["transcript"]:
                transcript = self.krn_parser.convert(text=text)
                vocab.extend(transcript)
        vocab = sorted(set(vocab))

        w2i = {}
        i2w = {}
        for i, w in enumerate(vocab):
            w2i[w] = i + 1
            i2w[i + 1] = w
        w2i["<PAD>"] = 0
        i2w[0] = "<PAD>"

        return w2i, i2w

    def check_and_retrieve_max_lens(self):
        max_lens = {}

        if os.path.isfile(self.max_lens_path):
            with open(self.max_lens_path, "r") as file:
                max_lens = json.load(file)
        else:
            max_lens = self.make_max_lens()
            with open(self.max_lens_path, "w") as file:
                json.dump(max_lens, file)

        return max_lens

    def make_max_lens(self):
        # Set the maximum lengths for the whole QUARTETS collection:
        # 1) Get the maximum transcript length
        # 2) Get the maximum audio length
        # 3) Get the frame multiplier factor so that
        # the frames input to the RNN are equal to the
        # length of the transcript, ensuring the CTC condition
        max_seq_len = 0
        max_audio_len = 0
        max_frame_multiplier_factor = 0

        full_ds = load_dataset("PRAIG/quartets-quartets")
        for split in SPLITS:
            for sample in full_ds[split]:
                # Max transcript length
                transcript = self.krn_parser.convert(text=sample["transcript"])
                max_seq_len = max(max_seq_len, len(transcript))

                # Max audio length
                audio = preprocess_audio(
                    raw_audio=sample["audio"]["array"],
                    sr=sample["audio"]["sampling_rate"],
                    dtype=torch.float32,
                )
                max_audio_len = max(max_audio_len, audio.shape[2])
                # Max frame multiplier factor
                max_frame_multiplier_factor = max(
                    max_frame_multiplier_factor,
                    math.ceil(((2 * len(transcript)) + 1) / audio.shape[2]),
                )

        return {
            "max_seq_len": max_seq_len,
            "max_audio_len": max_audio_len,
            "max_frame_multiplier_factor": max_frame_multiplier_factor,
        }
