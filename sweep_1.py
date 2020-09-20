from fastai.vision.all import *
from fastaudio.core.all import *
from fastaudio.augment.all import *
from fastcore.script import *

import wandb
from fastai.callback.wandb import *

from utils import * 


assert torch.cuda.is_available()


project_name = 'fastaudio-esc-50'

path = untar_data(URLs.ESC50)
audio_path = path/'audio'
meta_path = path/'meta'

df = pd.read_csv(meta_path/'esc50.csv')
fold_num = 1

config = dict(
    sample_rate=44100,
    n_fft=8192,
    n_mels=256,
    hop_length=256,
    win_length=1764,
    n_epochs=20,
    batch_size=32,
    mix_up=0.4,
    arch='resnet34',
)

tags = [
    'mel_spec',
]

wandb.init(
    config=config,
    project=project_name, 
    tags=tags,
)

config = wandb.config
    
audio_config = AudioConfig.BasicMelSpectrogram(
    sample_rate=config.sample_rate,
    n_fft=config.n_fft,
    win_length=config.win_length,
    hop_length=config.hop_length,
    normalized=True,
    n_mels=config.n_mels,
)

audio_to_spec = AudioToSpec.from_cfg(audio_config)
audio_block = AudioBlock(sample_rate=config.sample_rate)
dblock = DataBlock(
    blocks=(audio_block, CategoryBlock),  
    get_x=ColReader("filename", pref=audio_path),
    splitter=CrossValidationSplitter(fold=fold_num),
    batch_tfms = [audio_to_spec],
    get_y=ColReader("category")
)

dls = dblock.dataloaders(df, bs=config.batch_size)

arch = eval(config.arch)
learn = cnn_learner(
    dls, arch, 
    config=cnn_config(n_in=1),
    loss_fn=CrossEntropyLossFlat,
    metrics=accuracy
)

learn.to_fp16()

wandb_cb = WandbCallback(log_model=False, log_preds=False)
cbs = [wandb_cb, MixUp(config.mix_up)]

lr_min, lr_steep = learn.lr_find()
learn.fine_tune(config.n_epochs, base_lr=lr_min, cbs=cbs)

wandb.join()