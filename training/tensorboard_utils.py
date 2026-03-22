from typing import Any, Dict

from logging import getLogger
import warnings
from matplotlib.figure import Figure
import torch
import matplotlib.pyplot as plt
from torchaudio.transforms import MelSpectrogram
from torch.utils.tensorboard import SummaryWriter
from dataset.speechcommands import idx2name

import numpy as np
from pytorch_lightning.loggers import TensorBoardLogger
import itertools
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import confusion_matrix
import re



logger = getLogger(__name__)


def log_mel_spectrogram(
    writer: SummaryWriter,
    name: str,
    log_mel: torch.Tensor,
    label: str,
    step: int,
    predicted: str=None,
    x_axis: str = 'Time (s)',
    y_axis: str = 'Mels',
    cmap: str = 'viridis',
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    im = ax.imshow(log_mel.cpu().numpy(), aspect="auto", origin="lower", cmap=cmap)
    fig.colorbar(im, fraction=0.04, pad=0.01)
    title = f"Label: {label}"
    if predicted is not None:
        title += f" | Predicted: {predicted}"
    ax.set_title(title)
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    image = render_figure(fig)
    writer.add_image(f"{name}", image, step)


def plot_confusion_matrix(correct_labels,
                          predict_labels,
                          labels,
                          title='Confusion matrix',
                          normalize=False):
    ''' 
    Parameters:
        correct_labels                  : These are your true classification categories.
        predict_labels                  : These are you predicted classification categories
        labels                          : This is a lit of labels which will be used to display the axix labels
        title='Confusion matrix'        : Title for your matrix
        tensor_name = 'MyFigure/image'  : Name for the output summay tensor
    Returns:
        summary: TensorFlow summary 
    Other itema to note:
        - Depending on the number of category and the data , you may have to modify the figzie, font sizes etc. 
        - Currently, some of the ticks dont line up due to rotations.
    '''
    cm = confusion_matrix(correct_labels, predict_labels, labels=range(len(labels)))
    if normalize:
        cm = cm.astype('float')*10 / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm, copy=True)
        cm = cm.astype('int')

    np.set_printoptions(precision=2)

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5), dpi=160, facecolor='w', edgecolor='k')
    fig.suptitle('confusion_matrix',fontsize=7)
    im = ax.imshow(cm, cmap='Oranges')

    classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in labels]
    #classes = ['\n'.join(l) for l in classes]

    tick_marks = np.arange(len(classes))

    ax.set_xlabel('Predicted', fontsize=7)
    ax.set_xticks(tick_marks)
    c = ax.set_xticklabels(classes, fontsize=5, rotation=0,  ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label', fontsize=7)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=5, va ='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd') if cm[i,j]!=0 else '.', horizontalalignment="center", fontsize=6, verticalalignment='center', color= "black")
    fig.set_tight_layout(True)

    return fig

def log_confusion_matrix(writer: SummaryWriter,
                         name: str,
                         correct_labels,
                         predict_labels,
                         labels,
                         title='Confusion matrix',
                         normalize=False,
                         step: int = 0):
    fig = plot_confusion_matrix(correct_labels, predict_labels, labels, title, normalize)
    image = render_figure(fig)
    writer.add_image(f"{name}", image, step)
    return fig

def barplot(result_dict, title, figsize=(4,12), minor_interval=0.2, log=False):
    fig, ax = plt.subplots(1,1, figsize=figsize)
    metric = {}
    for idx, item in enumerate(result_dict[None][title]):
        metric[idx2name[idx]] = item
    xlabels = list(metric.keys())
    values = list(metric.values())
    if log:
        values = np.log(values)
    ax.barh(xlabels, values)
    ax.tick_params(labeltop=True, labelright=False)
    ax.xaxis.grid(True, which='minor')
    ax.xaxis.set_minor_locator(MultipleLocator(minor_interval))
    ax.set_ylim([-1,len(xlabels)])
    ax.set_title(title)
    ax.grid(axis='x')
    ax.grid(visible=True, which='minor', linestyle='--')
    # fig.savefig(f'{title}.png', bbox_inches='tight')
    fig.tight_layout() # prevent edge from missing
    return fig

def log_barplot(writer: SummaryWriter,
                name: str,
                result_dict,
                title,
                figsize=(4,12),
                minor_interval=0.2,
                log=False,
                step: int = 0):
    fig = barplot(result_dict, title, figsize, minor_interval, log)
    image = render_figure(fig)
    writer.add_image(f"{name}", image, step)
    return fig


def render_figure(fig: Figure) -> np.ndarray:
    with warnings.catch_warnings():
        # Squash UserWarning for missing from korean/Japanese/Chineses font.
        # ex. UserWarning: Glyph 4369 (\N{HANGUL CHOSEONG PHIEUPH}) missing from current font.
        warnings.filterwarnings("ignore", category=UserWarning)
        if not fig.get_constrained_layout():
            fig.tight_layout()
        try:
            fig.canvas.draw()
        except Exception as e:
            logger.exception(f"Could not draw figure {e}")
            del fig
            return np.zeros(())
        # tostring_rgb() was removed from some backends (e.g. macOS); buffer_rgba is the supported API.
        canvas = fig.canvas
        if hasattr(canvas, "buffer_rgba"):
            rgba = np.asarray(canvas.buffer_rgba())
            img = np.ascontiguousarray(rgba[..., :3])
        else:
            img = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape(
                canvas.get_width_height()[::-1] + (3,)
            )
        img = img / 255.0
        img = np.transpose(img, (2, 0, 1))
        plt.close(fig)
        del fig
    return img
