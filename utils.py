from fastai.vision.all import * 
from fastaudio.core.all import *
from fastaudio.augment.all import *

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
    
def pitch_shift(t: AudioTensor, n_steps):
    assert t.shape[0] == 1
    x = t.numpy()[0]
    y = librosa.effects.pitch_shift(x, t.sr, n_steps)
    t.data = torch.from_numpy(y[np.newaxis, :])
    return t

class PitchShifter(RandTransform):
    """Randomly shifts the audio signal by `max_pct` %.
        direction must be -1(left) 0(bidirectional) or 1(right).
    """

    def __init__(self, max_steps=3):
        super().__init__()
        self.max_steps = max_steps

    def before_call(self, b, split_idx):
        super().before_call(b, split_idx)
        self.shift_factor = random.uniform(-1, 1)

    def encodes(self, t: AudioTensor):
        n_steps = self.shift_factor * self.max_steps
        t = pitch_shift(t, n_steps)
        return t