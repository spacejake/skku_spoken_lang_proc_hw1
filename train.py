# import related libraries
import argparse
import warnings

import torch
from torch import Tensor
import torchaudio 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import WeightedRandomSampler, DataLoader
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer


from dataset.speechcommands import SPEECHCOMMANDS_12C, collate_speechcommands_batch  # 12 classes KWS
from dataset.augmentations import TimeMask
from model.speechcommand import ConvNextASR

# PyTorch MPS stft emits a noisy deprecation warning about internal `out` resize (harmless for training).
warnings.filterwarnings(
    "ignore",
    message=r"An output with one or more elements was resized since it had shape \[\], which does not",
    category=UserWarning,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ConvNextASR on Speech Commands (12-class KWS).")
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
        help="Root directory for the dataset (data under <data-root>/SpeechCommands/...). Default: './data'.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download Speech Commands v0.02 via torchaudio if files are missing (default: do not download).",
    )
    args = parser.parse_args()
    data_root = args.data_root
    download_option = args.download
    folder_in_archive = "./SpeechCommands"

    # Get available devices
    if torch.cuda.is_available():
        device = 'cuda:0'
        print("CUDA device found. Using GPU.")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("MPS device found. Using GPU.")
    else:
        device = torch.device("cpu")
        print("MPS device not found. Using CPU.")

    gpus = 1
    batch_size = 100
    max_epochs = 200
    check_val_every_n_epoch = 2
    num_sanity_val_steps = 5
    output_dim = 12

    trainset = SPEECHCOMMANDS_12C(
        root=data_root,
        url='speech_commands_v0.02',
        folder_in_archive=folder_in_archive,
        download=download_option,
        subset='training',
        # subset='validation',
        transform=TimeMask(p=0.5)
    )

    validset = SPEECHCOMMANDS_12C(
        root=data_root,
        url='speech_commands_v0.02',
        folder_in_archive=folder_in_archive,
        download=download_option,
        subset='validation',
    )

    #data rebalancing
    class_weights = [1,1,1,1,1,1,1,1,1,1,4.6,1/17]
    sample_weights = [0] * len(trainset)
    for idx, (waveform, rate, label, speaker_id, utterance_number, original_label) in enumerate(trainset):
        class_weight = class_weights[label]
        sample_weights[idx] = class_weight
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    trainloader = DataLoader(
        trainset,
        collate_fn=collate_speechcommands_batch,
        batch_size=batch_size,
        sampler=sampler,
    )

    validloader = DataLoader(
        validset,
        collate_fn=collate_speechcommands_batch,
        batch_size=batch_size,
    )

    model = ConvNextASR(max_epochs=max_epochs)
    model = model.to(device)

    trainer = Trainer(
        max_epochs=max_epochs,
        check_val_every_n_epoch=check_val_every_n_epoch,
        num_sanity_val_steps=num_sanity_val_steps,
        logger=TensorBoardLogger(save_dir="lightning_logs", name="speech_commands"),
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(
                monitor="Validation/acc",
                mode="max",
                save_top_k=1,
                filename="best-val-acc-epoch{epoch:03d}",
                save_weights_only=False,
                verbose=True,
            ),
        ],
    )

    trainer.fit(model, trainloader, validloader)
