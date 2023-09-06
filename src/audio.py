import numpy as np
from madmom.audio import SignalProcessor, FramedSignalProcessor, LogarithmicFilteredSpectrogramProcessor
from madmom.processors import Processor

class Spectrogram():
    def __init__(self, spec, params_dict=None):
        self.spec = spec
        if params_dict is not None:
            self.params_dict = params_dict

    @property
    def params(self):
        return self.params_dict


class MadmomAudioProcessor(Processor):
    def __init__(self,
                 sr=22050,
                 hop_size=None,
                 fps=None,
                 frame_size=2048,
                 num_bands_oct=24):
        self.sr = sr
        self.n_fft = frame_size
        self.num_bands_oct = num_bands_oct

        if fps is None:
            self.hop_size = 512
        else:
            self.hop_size = sr // fps

        self.sig_proc = SignalProcessor(num_channels=1, sample_rate=sr, norm=True)
        self.fsig_proc = FramedSignalProcessor(frame_size=frame_size, hop_size=hop_size, fps=fps, origin='future')
        self.spec_proc = LogarithmicFilteredSpectrogramProcessor(num_bands=num_bands_oct, fmin=20, fmax=16000)
        self.name = "madmom" + '_sr=' + str(sr) + '_nfft=' + str(self.n_fft) + '_hop=' + str(int(self.hop_size))

    def process(self, file_path, **kwargs):
        sig = np.trim_zeros(self.sig_proc.process(file_path))
        fsig = self.fsig_proc.process(sig)
        spec = self.spec_proc.process(fsig)
        return Spectrogram(spec.transpose(), params_dict=self.get_params)

    def times_to_frames(self, times):
        return np.floor(np.array(times) * self.sr / self.hop_size).astype(int)

    def frames_to_times(self, frames):
        return frames * self.hop_size / self.sr

    @property
    def get_params(self):
        param_dict = {"sr": self.sr,
                      "n_fft": self.n_fft,
                      "hop_length": self.hop_size,
                      "n_bands_octave": self.num_bands_oct,
                      "name": self.name}
        return param_dict
