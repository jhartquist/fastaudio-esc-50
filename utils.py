from fastai.vision.all import * 
from fastaudio.core.all import *

# from https://github.com/fastaudio/Audio-Competition/blob/master/ESC-50-baseline-1Fold.ipynb
def CrossValidationSplitter(col='fold', fold=1):
    "Split `items` (supposed to be a dataframe) by fold in `col`"
    def _inner(o):
        assert isinstance(o, pd.DataFrame), "ColSplitter only works when your items are a pandas DataFrame"
        col_values = o.iloc[:,col] if isinstance(col, int) else o[col]
        valid_idx = (col_values == fold).values.astype('bool')
        return IndexSplitter(mask2idxs(valid_idx))(o)
    return _inner


def get_audio_config(sample_rate: int, 
                     n_fft: int, 
                     win_length: int, 
                     hop_length: int, 
                     normalized: bool = False,
                     win_name: str = 'hann', 
                     n_mels: Optional[int] = None):
    window_fn = get_window_fn(win_name)
    if n_mels is not None:
        return AudioConfig.BasicMelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            window_fn=window_fn,
            hop_length=hop_length,
            normalized=normalized,
            n_mels=n_mels,
        )
    else:
        return AudioConfig.BasicSpectrogram(
            n_fft=n_fft,
            win_length=win_length,
            window_fn=window_fn,
            hop_length=hop_length,
            normalized=normalized,
        )

def get_data_block(audio_path, 
                   sample_rate,
                   fold_num, 
                   audio_config, 
                   signal_tfms):
    audio_to_spec = AudioToSpec.from_cfg(audio_config)
    audio_block = AudioBlock(sample_rate=sample_rate)
    dblock = DataBlock(blocks=(audio_block, CategoryBlock),  
                       get_x=ColReader("filename", pref=audio_path),
                       splitter=CrossValidationSplitter(fold=fold_num),
                       item_tfms=signal_tfms,
                       batch_tfms = [audio_to_spec],
                       get_y=ColReader("category"))
    return dblock


def get_learner(data, arch, mix_up = None):
    return cnn_learner(data,
                       arch, 
                       config=cnn_config(n_in=1),
                       loss_fn=CrossEntropyLossFlat,
                       metrics=accuracy)
