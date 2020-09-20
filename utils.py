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

class StackedMelSpecs(Transform):
    "Stacks Mel spectrograms with different resolutions into a single image."
    
    def __init__(self, n_fft, n_mels, sample_rate, win_lengths, hop_length):
        store_attr()
        # mel spectrum extractors
        assert max(win_lengths) <= n_fft
        self.specs = [AudioToSpec.from_cfg(
            AudioConfig.BasicMelSpectrogram(n_fft=n_fft,
                                            hop_length=hop_length,
                                            win_length=win_length,
                                            normalized=True,
                                            n_mels=n_mels,
                                            sample_rate=sample_rate)) 
                      for win_length in win_lengths]

    def encodes(self, x: AudioTensor) -> AudioSpectrogram:       
        return AudioSpectrogram(torch.cat([spec(x) for spec in self.specs], axis=1))