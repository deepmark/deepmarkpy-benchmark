import numpy as np

from deepmarkpy.core.base_attack import BaseAttack

class EqualizerAttack(BaseAttack):

    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        """
        This attacks performs an equalizer attack on an audio signal, that's been implemented here: https://github.com/chenwj1989/pafx/tree/main.
        This is a 10-band equalizer with frequencies from 31Hz to 16kHz with the distance of 1 octave. 
        Args:
            audio (np.ndarray): The input audio signal.
            **kwargs: Additional parameters for the equalizer attack:
                - sampling_rate (int): The sampling rate of the audio signal in Hz (required).
                - gains (np.ndarray): Specifies how much to boost or cut each of the 10 bands.
        Returns:
            np.ndarray: The processed audio signal.

        Raises:
            ValueError: If the `sampling_rate` is not provided in `kwargs`.

        """

        sampling_rate = kwargs.get("sampling_rate", None)
        gains = kwargs.get("gains", self.config.get("gains"))

        center_freqs = [31.5, 63, 125, 250, 500, 
                1000, 2000, 4000, 8000, 16000] #add 8000, 16000 for higher sample rates

        band_widths = [22, 44, 88, 177, 355, 710, 
               1420, 2840, 5680, 11360]
        
        self.filters = []
        self.filters.append(self.Biquad(sampling_rate, 'LowShelf', center_freqs[0],
                                    band_widths[0], gains[0]))  
        for i in range(1, 8 - 1):
            self.filters.append(self.Biquad(sampling_rate, 'Peaking', center_freqs[i],
                                       band_widths[i], gains[i]))   

        self.filters.append(self.Biquad(sampling_rate, 'HighShelf',
                                    center_freqs[8 - 1],
                                    band_widths[8- 1],
                                    gains[8 - 1])
                            )  

        filtered=np.zeros_like(audio)
        for i in range(len(audio)):
            out = audio[i]
            for filter in self.filters:
                out += filter.process(out)
            filtered[i]=out

        filtered = filtered / max(np.abs(filtered))
        return filtered

        

    class Biquad():
        def __init__(
            self,
            sample_rate: float,
            filter_type: str = None,
            fc: float = 1000.0,
            bandwidth: float = 1.0,
            gain_db: float = 1.0
        ) -> None:
            """
            Implements a biquad IIR filter for common audio filter types.
            Args:
                sample_rate (float): Sampling rate of the signal.
                filter_type (str): Type of the filter. Options:
                    "LowPass", "HighPass", "BandPass", "AllPass", "Notch",
                    "Peaking", "LowShelf", "HighShelf".
                fc (float): Cutoff/center frequency in Hz.
                bandwidth (float): Bandwidth of the filter in Hz.
                gain_db (float): Gain in dB for peaking and shelving filters.
            Raises:
                ValueError: For invalid sample_rate, fc, or filter_type values.
            """
            if sample_rate < 0.0:
                raise ValueError("sample_rate cannot be given a negative value")
            self.sample_rate = sample_rate
        
            self.b = np.zeros(3)
            self.a = np.zeros(3)
            self.a [0] = 1.0
            self.b[0] = 1.0

            self.y = None
            self.x_buf = np.zeros(2)
            self.y_buf = np.zeros(2)
                
            self.filter_type = filter_type

            if fc < 0.0 or fc >= self.sample_rate / 2.0:
                raise ValueError(f"illegal value: fc={fc}")
            self._fc = fc

            self._gain_db = gain_db
            A = 10.0 ** (gain_db/40.0)
            A_add_1 = A + 1.0
            A_sub_1 = A - 1.0
            sqrt_A  = np.sqrt(A)

            w0 = 2.0 * np.pi * self._fc / self.sample_rate
            cos_w0 = np.cos(w0) 
            sin_w0 = np.sin(w0)
            alpha = 0.5 * sin_w0 * fc / bandwidth

            if filter_type == "LowPass":
                self.b[0] = (1.0 - cos_w0) * 0.5
                self.b[1] = (1.0 - cos_w0)
                self.b[2] = (1.0 - cos_w0) * 0.5
                self.a[0] =  1.0 + alpha
                self.a[1] = -2.0 * cos_w0
                self.a[2] =  1.0 - alpha

            elif filter_type == "HighPass":
                self.b[0] =  (1.0 + cos_w0) * 0.5
                self.b[1] = -(1.0 + cos_w0)
                self.b[2] =  (1.0 + cos_w0) * 0.5
                self.a[0] =   1.0 + alpha
                self.a[1] =  -2.0 * cos_w0
                self.a[2] =   1.0 - alpha

            elif filter_type == "BandPass":
                self.b[0] =  alpha
                self.b[1] =  0.0
                self.b[2] = -alpha
                self.a[0] =  1.0 + alpha
                self.a[1] = -2.0 * cos_w0
                self.a[2] =  1.0 - alpha

            elif filter_type == "AllPass":
                self.b[0] =  1.0 - alpha
                self.b[1] = -2.0 * cos_w0
                self.b[2] =  1.0 + alpha
                self.a[0] =  1.0 + alpha
                self.a[1] = -2.0 * cos_w0
                self.a[2] =  1.0 - alpha

            elif filter_type == "Notch":
                self.b[0] =  1.0
                self.b[1] = -2.0 * cos_w0
                self.b[2] =  1.0
                self.a[0] =  1.0 + alpha
                self.a[1] = -2.0 * cos_w0
                self.a[2] =  1.0 - alpha

            elif filter_type == "Peaking":
                if A != 1.0:
                    self.b[0] =  1.0 + alpha * A
                    self.b[1] = -2.0 * cos_w0
                    self.b[2] =  1.0 - alpha * A
                    self.a[0] =  1.0 + alpha / A
                    self.a[1] = -2.0 * cos_w0
                    self.a[2] =  1.0 - alpha / A

            elif filter_type == "LowShelf":
                if A != 1.0:
                    self.b[0] =     A * (A_add_1 - A_sub_1 * cos_w0 + 2 * sqrt_A * alpha)
                    self.b[1] = 2 * A * (A_sub_1 - A_add_1 * cos_w0)
                    self.b[2] =     A * (A_add_1 - A_sub_1 * cos_w0 - 2 * sqrt_A * alpha)
                    self.a[0] =          A_add_1 + A_sub_1 * cos_w0 + 2 * sqrt_A * alpha
                    self.a[1] =    -2 * (A_sub_1 + A_add_1 * cos_w0)
                    self.a[2] =          A_add_1 + A_sub_1 * cos_w0 - 2 * sqrt_A * alpha

            elif filter_type == "HighShelf":
                if A != 1.0:
                    self.b[0] =      A * (A_add_1 + A_sub_1 * cos_w0 + 2 * sqrt_A * alpha)
                    self.b[1] = -2 * A * (A_sub_1 + A_add_1 * cos_w0)
                    self.b[2] =      A * (A_add_1 + A_sub_1 * cos_w0 - 2 * sqrt_A * alpha)
                    self.a[0] =           A_add_1 - A_sub_1 * cos_w0 + 2 * sqrt_A * alpha
                    self.a[1] =      2 * (A_sub_1 - A_add_1 * cos_w0)
                    self.a[2] =           A_add_1 - A_sub_1 * cos_w0 - 2 * sqrt_A * alpha
            else:
                raise ValueError(f"invalid filter_type: {filter_type}")
            
            self.b /= self.a[0]
            self.a /= self.a[0]
            
        def process(self, x: float) -> float:
            """
            Processes a single input sample using the biquad filter.
            Args:
                x (float): Input audio sample.
            Returns:
                float: Filtered output sample.
            """
            y = (
                self.b[0] * x +
                self.b[1] * self.x_buf[1] +
                self.b[2] * self.x_buf[0] -
                self.a[1] * self.y_buf[1] -
                self.a[2] * self.y_buf[0]
            )

            self.x_buf[0] = self.x_buf[1]
            self.x_buf[1] = x
            self.y_buf[0] = self.y_buf[1]
            self.y_buf[1] = y

            return y
