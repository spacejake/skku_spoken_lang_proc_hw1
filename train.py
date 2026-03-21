
# import related libraries
import warnings

import torch
from torch import Tensor
import torchaudio 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import WeightedRandomSampler, DataLoader
import torch.optim as optim
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt


from dataset.speechcommands import SPEECHCOMMANDS_12C, collate_speechcommands_batch, idx2name  # 12 classes KWS
from dataset.augmentations import TimeMask
from model.asr import SpeechCommandClassifier
from model.encoder import Encoder
from training.tensorboard_utils import log_mel_spectrogram

# PyTorch MPS stft emits a noisy deprecation warning about internal `out` resize (harmless for training).
warnings.filterwarnings(
    "ignore",
    message=r"An output with one or more elements was resized since it had shape \[\], which does not",
    category=UserWarning,
)

# tb_logger = TensorBoardLogger(log_dir='./logs')

class SpeechCommand(LightningModule):
    @rank_zero_only
    def log_spec_audio(self, name, spec, waveforms, labels, step, preds=None,):
        writer = self.logger.experiment
        step = self.global_step  # or self.current_epoch
        # spec: e.g. [B, C, T] — pick one example, normalize to [0, 1] for display
        log_mel_spectrogram(
            self.logger.experiment,
            name=name,
            log_mel=spec[0],
            label=idx2name[int(labels[0].item())],
            predicted=idx2name[int(preds[0].item())] if preds is not None else None,
            step=step
        )

        # Audio: shape (1, num_samples), float in [-1, 1]
        w = waveforms[0].detach().cpu()#.clamp(-1, 1)
        if w.dim() == 1:
            w = w.unsqueeze(0)
        writer.add_audio("train/waveform", w, step, sample_rate=16000)
        return

    def training_step(self, batch, batch_idx):
        outputs, spec = self(batch['waveforms']) 
        #return outputs [2D] for calculate loss, return spec [3D] for visual
        loss = self.criterion(outputs, batch['labels'].long())
        preds = outputs.argmax(-1)
        acc = sum(preds == batch['labels'])/outputs.shape[0] #batch wise
        
        self.log('Train/acc', acc, on_step=False, on_epoch=True)
        self.log('Train/Loss', loss, on_step=False, on_epoch=True)
        if self.global_step % 5000 == 0:
            self.log_spec_audio(
                name="train/spec", 
                spec=spec, 
                waveforms=batch['waveforms'].detach().cpu(), 
                labels=batch['labels'].detach().cpu(),
                preds=preds.detach().cpu(),
                step=self.global_step
            )
        return loss

    def on_before_optimizer_step(self, optimizer):
        # Total L2 norm of gradients (same quantity as clip_grad_norm_ returns).
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), float("inf"))
        self.log(
            "Train/grad_norm",
            grad_norm,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
        )

     
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        optimizer.step(closure=optimizer_closure)
        # after optimizer step, clamp mel basis if model has trainable mel layer
        if hasattr(self, 'mel_layer') and hasattr(self.mel_layer, 'mel_basis'):
            with torch.no_grad():
                torch.clamp_(self.mel_layer.mel_basis, 0, 1)


    def validation_step(self, batch, batch_idx):               
        outputs, spec = self(batch['waveforms'])
        preds = outputs.argmax(-1)
        loss = self.criterion(outputs, batch['labels'].long())        

        self.log('Validation/Loss', loss, on_step=False, on_epoch=True)
        if not hasattr(self, '_val_outputs'):
            self._val_outputs = []
        self._val_outputs.append((outputs.detach(), batch['labels'].detach()))
        
        if self.global_step % 1000 == 0:
            self.log_spec_audio(
                name="train/spec", 
                spec=spec, 
                waveforms=batch['waveforms'].detach().cpu(), 
                labels=batch['labels'].detach().cpu(),
                preds=preds.detach().cpu(),
                step=self.global_step
            )
        return loss

    
    def on_validation_epoch_end(self):
        if not hasattr(self, '_val_outputs') or len(self._val_outputs) == 0:
            return
        pred = []
        label = []
        pred = torch.cat([x[0] for x in self._val_outputs], 0)
        label = torch.cat([x[1] for x in self._val_outputs], 0)
        acc = (pred.argmax(-1) == label).float().mean()
        self.log('Validation/acc', acc, on_step=False, on_epoch=True)
        self._val_outputs.clear()  

    
    def configure_optimizers(self):
        model_param = []
        for name, params in self.named_parameters():
            if 'mel_layer.' in name:
                pass
            else:
                model_param.append(params)          

        optimizer = optim.AdamW(model_param, lr=1e-3, weight_decay=0.001, betas=(0.9, 0.95))
        max_epochs = getattr(self, "max_epochs", 200)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs, eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

class ConvNextASR(SpeechCommand):
    def __init__(self, max_epochs: int = 200):
        super().__init__()
        self.max_epochs = max_epochs
        self.model = SpeechCommandClassifier(
            num_classes=12,
            encoder=Encoder(),
            odim=128,
            head_dropout=0.2,
            pooling="mean",
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        out, spec = self.model(x, return_intermediates=True)

        #out: 2D [B, 12]
        return out, spec



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

    trainset = SPEECHCOMMANDS_12C(
        root=data_root,
        url='speech_commands_v0.02',
        folder_in_archive='./data/SpeechCommands',
        download= download_option,
        subset='training',
        # subset='validation',
        transform=TimeMask(p=0.5)
    )

    validset = SPEECHCOMMANDS_12C(
        root=data_root,
        url='speech_commands_v0.02',
        folder_in_archive='./data/SpeechCommands',
        download= download_option,
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

