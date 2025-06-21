import gc

import fire
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger

from networks.crnn.model import CTCTrainedCRNN
from networks.transformer.model import A2STransformer
from my_utils.ctc_dataset import CTCDataModule
from my_utils.ar_dataset import ARDataModule
from my_utils.seed import seed_everything

seed_everything(42, benchmark=False)


def train(
    ds_name,
    model_type: str = "crnn",
    attn_window: int = -1,
    use_voice_change_token: bool = False,
    epochs: int = 1000,
    patience: int = 20,
    batch_size: int = 16,
    check_val_every_n_epoch: int = 5,
):
    gc.collect()
    torch.cuda.empty_cache()

    # Experiment info
    print("TRAIN EXPERIMENT")
    print(f"\tDataset: {ds_name}")
    print(f"\tModel type: {model_type}")
    print(f"\tAttention window: {attn_window} (Used if model type is transformer)")
    print(f"\tUse voice change token: {use_voice_change_token}")
    print(f"\tEpochs: {epochs}")
    print(f"\tPatience: {patience}")
    print(f"\tBatch size: {batch_size}")
    print(f"\tCheck Val Every N epoch: {check_val_every_n_epoch}")

    if model_type == "crnn":
        # Data module
        datamodule = CTCDataModule(
            ds_name=ds_name,
            use_voice_change_token=use_voice_change_token,
            batch_size=batch_size,
        )
        datamodule.setup(stage="fit")
        w2i, i2w = datamodule.get_w2i_and_i2w()

        # Model
        model = CTCTrainedCRNN(
            w2i=w2i,
            i2w=i2w,
            max_audio_len=datamodule.get_max_audio_len(),
            frame_multiplier_factor=datamodule.get_frame_multiplier_factor(),
        )
        # Override the datamodule width reduction factors with that of the model
        datamodule.width_reduction = model.width_reduction

    elif model_type == "transformer":
        # Data module
        datamodule = ARDataModule(
            ds_name=ds_name,
            use_voice_change_token=use_voice_change_token,
            batch_size=batch_size,
        )
        datamodule.setup(stage="fit")
        w2i, i2w = datamodule.get_w2i_and_i2w()

        # Model
        model = A2STransformer(
            max_seq_len=datamodule.get_max_seq_len(),
            max_audio_len=datamodule.get_max_audio_len(),
            w2i=w2i,
            i2w=i2w,
            attn_window=attn_window,
            teacher_forcing_prob=0.2,
        )

    else:
        print(f"Model type {model_type} not implemented")
        raise NotImplementedError

    # Train, validate and test
    callbacks = [
        ModelCheckpoint(
            dirpath=f"weights/{model_type}" if not use_voice_change_token else f"weights/{model_type}-VCT",
            filename=ds_name,
            monitor="val_sym-er",
            verbose=True,
            save_last=False,
            save_top_k=1,
            save_weights_only=False,
            mode="min",
            auto_insert_metric_name=False,
            every_n_epochs=5,
            save_on_train_epoch_end=False,
        ),
        EarlyStopping(
            monitor="val_sym-er",
            min_delta=0.01,
            patience=patience,
            verbose=True,
            mode="min",
            strict=True,
            check_finite=True,
            divergence_threshold=100.00,
            check_on_train_epoch_end=False,
        ),
    ]
    trainer = Trainer(
        logger=WandbLogger(
            project="A2S-Poly-ICASSP",
            group=f"{model_type}" if not use_voice_change_token else f"{model_type}-VCT",
            name=f"Train-{ds_name}_Test-{ds_name}",
            log_model=False,
        ),
        callbacks=callbacks,
        max_epochs=epochs,
        check_val_every_n_epoch=check_val_every_n_epoch,
        deterministic=False,  # If True, raises error saying that CTC loss does not have this behaviour
        benchmark=False,
        precision="16-mixed",  # Mixed precision training
    )
    trainer.fit(model, datamodule=datamodule)
    if model_type == "crnn":
        model = CTCTrainedCRNN.load_from_checkpoint(callbacks[0].best_model_path)
    else:
        model = A2STransformer.load_from_checkpoint(callbacks[0].best_model_path)
    model.freeze()
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    fire.Fire(train)
