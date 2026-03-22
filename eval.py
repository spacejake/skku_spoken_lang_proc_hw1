
# import related libraries
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
    data_root = './' # Download the data here
    download_option = False
    output_dim = 12

    testset = SPEECHCOMMANDS_12C(
        root=data_root,
        url='speech_commands_v0.02',
        folder_in_archive='./data/SpeechCommands',
        download= download_option,
        subset='testing',
    )

    testloader = DataLoader(
        testset,
        collate_fn=collate_speechcommands_batch,
        batch_size=batch_size,
        num_workers=1,
    )

    model = ConvNextASR.load_from_checkpoint('lightning_logs/speech_commands/v1/checkpoints/best-val-acc-epochepoch=173.ckpt')
    model = model.to(device)


    trainer = Trainer(
        max_epochs=max_epochs,
        logger=TensorBoardLogger(save_dir="lightning_logs", name="eval_speech_commands"),
        check_val_every_n_epoch= check_val_every_n_epoch,
        num_sanity_val_steps=num_sanity_val_steps
    )

    trainer.test(model, testloader)
