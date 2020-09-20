from fastai.vision.all import *
from fastaudio.core.all import *
from fastaudio.augment.all import *
from fastcore.script import *
from fastai.distributed import *

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
    n_fft=4096,
    n_mels=128,
    hop_length=220,
    n_epochs=150,
    batch_size=16,
    mix_up=0.8,
    arch='densenet201',
)

win_lengths = [441, 1103, 2205] # 10ms, 20ms, 80ms

tags = [
    'mel_spec',
]

wandb.init(
    config=config,
    project=project_name, 
    tags=tags,
)

config = wandb.config
    

audio_to_spec = StackedMelSpecs(
    n_fft=config.n_fft,
    n_mels=config.n_mels,
    sample_rate=config.sample_rate,
    win_lengths=win_lengths,
    hop_length=config.hop_length,
)
    
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
    loss_fn=CrossEntropyLossFlat,
    metrics=accuracy
)

learn.to_fp16()
learn.to_parallel([0,1])

wandb_cb = WandbCallback(log_model=False, log_preds=False)
cbs = [wandb_cb, MixUp(config.mix_up)]

lr_min, lr_steep = learn.lr_find()
learn.fine_tune(config.n_epochs, base_lr=lr_min, cbs=cbs)

wandb.join()
