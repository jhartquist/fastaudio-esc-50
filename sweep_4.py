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

config = dict(
    sample_rate=44100,
    n_fft=4096,
    n_mels=384,
    hop_length=441,
    win_length=2205,
    n_epochs=20,
    batch_size=32,
    arch='resnet18',
    learning_rate=1e-2,
    pitch_shift=True,
    signal_shift=True,
    add_noise=True,
    mix_up=True,
    trial=1,
)

tags = [
    'mel_spec',
]

group = f"{config['pitch_shift']}_{config['signal_shift']}_{config['add_noise']}_{config['mix_up']}"

wandb.init(
    config=config,
    project=project_name, 
    tags=tags,
    group=group,
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

signal_tfms = []

if config.pitch_shift:
    signal_tfms.append(PitchShifter())

if config.add_noise:
    signal_tfms.append(AddNoise())

if config.signal_shift:
    signal_tfms.append(SignalShifter())

audio_to_spec = AudioToSpec.from_cfg(audio_config)
audio_block = AudioBlock(sample_rate=config.sample_rate)
dblock = DataBlock(
    blocks=(audio_block, CategoryBlock),  
    get_x=ColReader("filename", pref=audio_path),
    splitter=CrossValidationSplitter(fold=1),
    batch_tfms = [audio_to_spec],
    item_tfms = signal_tfms,
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
cbs = [wandb_cb]
if config.mix_up:
    cbs.append(MixUp())

learn.fine_tune(config.n_epochs, 
                base_lr=config.learning_rate,
                cbs=cbs)

wandb.join()