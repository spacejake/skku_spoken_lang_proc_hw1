import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import rank_zero_only
from training.tensorboard_utils import log_mel_spectrogram

try:
    import model
except ImportError:
    import sys
    import git

    git_repo_root = git.Repo(".", search_parent_directories=True).git.rev_parse("--show-toplevel")
    if git_repo_root not in sys.path:
        sys.path.append(git_repo_root)

from dataset.speechcommands import idx2name, name2idx
from model.asr import SpeechCommandClassifier
from model.encoder import Encoder
from training.tensorboard_utils import log_mel_spectrogram, log_confusion_matrix, log_barplot
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns


class SpeechCommand(LightningModule):
    # Log mel + waveform every N batches within each val/test dataloader pass (0 disables).
    eval_log_spec_every_n_batches: int = 1000
    train_log_spec_every_n_batches: int = 10000

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
        prefix = name.rsplit("/", 1)[0] if "/" in name else name
        writer.add_audio(f"{prefix}/waveform", w, step, sample_rate=16000)
        return

    def training_step(self, batch, batch_idx):
        outputs, spec = self(batch['waveforms']) 
        #return outputs [2D] for calculate loss, return spec [3D] for visual
        loss = self.criterion(outputs, batch['labels'].long())
        preds = outputs.argmax(-1)
        acc = sum(preds == batch['labels'])/outputs.shape[0] #batch wise

        self.log('Train/acc', acc, on_step=False, on_epoch=True)
        self.log('Train/Loss', loss, on_step=False, on_epoch=True)
        n = self.train_log_spec_every_n_batches
        if n > 0 and batch_idx % n == 0:
            self.log_spec_audio(
                name="train/spec", 
                spec=spec, 
                waveforms=batch['waveforms'].detach().cpu(), 
                labels=batch['labels'].detach().cpu(),
                preds=preds.detach().cpu(),
                step=self.global_step
            )
        return loss

    def validation_step(self, batch, batch_idx):               
        outputs, spec = self(batch['waveforms'])
        preds = outputs.argmax(-1)
        loss = self.criterion(outputs, batch['labels'].long())        

        self.log('Validation/Loss', loss, on_step=False, on_epoch=True)
        if not hasattr(self, '_val_outputs'):
            self._val_outputs = []
        self._val_outputs.append((outputs.detach(), batch['labels'].detach()))

        n = self.eval_log_spec_every_n_batches
        if n > 0 and batch_idx % n == 0:
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

    def test_step(self, batch, batch_idx):               
        outputs, spec = self(batch['waveforms'])
        loss = self.criterion(outputs, batch['labels'].long())        

        self.log('Test/Loss', loss, on_step=False, on_epoch=True)          
        
        if not hasattr(self, '_test_outputs'):
            self._test_outputs = []
        self._test_outputs.append((outputs.detach(), batch['labels'].detach()))
    
        preds = outputs.argmax(-1)
        n = self.eval_log_spec_every_n_batches
        if n > 0 and batch_idx % n == 0:
            self.log_spec_audio(
                name="test/spec",
                spec=spec,
                waveforms=batch['waveforms'].detach().cpu(),
                labels=batch['labels'].detach().cpu(),
                preds=preds.detach().cpu(),
                step=self.global_step,
            )
        return loss

    def on_test_epoch_end(self):
        if not hasattr(self, '_test_outputs') or len(self._test_outputs) == 0:
            return
        pred = []
        label = []
        pred = torch.cat([x[0] for x in self._test_outputs], 0)
        label = torch.cat([x[1] for x in self._test_outputs], 0)
        
        result_dict = {}
        for key in [None, 'micro', 'macro', 'weighted']:
            result_dict[key] = {}
            p, r, f1, _ = precision_recall_fscore_support(
                label.cpu(),
                pred.argmax(-1).cpu(),
                average=key,
                zero_division=0
            )

            result_dict[key]['precision'] = p
            result_dict[key]['recall'] = r
            result_dict[key]['f1'] = f1
            
        log_barplot(self.logger.experiment, 'test/precision', result_dict, 'precision', figsize=(4,6))
        log_barplot(self.logger.experiment, 'test/recall', result_dict, 'recall', figsize=(4,6))
        log_barplot(self.logger.experiment, 'test/f1', result_dict, 'f1', figsize=(4,6))
            
        acc = sum(pred.argmax(-1) == label)/label.shape[0]
        self.log('test/acc', acc, on_step=False, on_epoch=True)
        
        self.log('test/micro_f1', result_dict['micro']['f1'], on_step=False, on_epoch=True)
        self.log('test/macro_f1', result_dict['macro']['f1'], on_step=False, on_epoch=True)
        self.log('test/weighted_f1', result_dict['weighted']['f1'], on_step=False, on_epoch=True)

        cm = log_confusion_matrix(
            self.logger.experiment,
            'test/confusion_matrix',
            label.cpu(),
            pred.argmax(-1).cpu(),
            name2idx.keys(),
            title='Test: Confusion matrix',
            normalize=False
        )
        return result_dict


class ConvNextASR(SpeechCommand):
    def __init__(self, max_epochs: int = 200, eval_log_spec_every_n_batches: int = 1000):
        super().__init__()
        self.max_epochs = max_epochs
        self.eval_log_spec_every_n_batches = eval_log_spec_every_n_batches
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
