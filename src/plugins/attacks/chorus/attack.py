import numpy as np
import math

from core.base_attack import BaseAttack

class ChorusAttack(BaseAttack):

    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        """
        This attack performs a chorus attack on an audio signal, that's been included in the StirMark Benchmark paper,
        but implemented slightly different (https://github.com/chenwj1989/pafx implementation).
        Args:
            audio (np.ndarray): The input audio signal.
            **kwargs: Additional parameters for the chorus attack:
                - sampling_rate (int): The sampling rate of the audio signal in Hz (required).
                - start_delays_chorus (float): Starting amount of time (in seconds) that the input signal is delayed before modulation.
                - w_delays_chorus (float): Controls how much the delay time is varied  by the sine wave.
                - delay_rates_chorus (float): Frequency (Hz) of the modulation sine wave — typically a low frequency (e.g., 0.3-5 Hz).
                - dry_gain_chorus (float): The amount of the original signal to keep (0.0 to 1.0).
                - chorus_gains_chorus (float): The amount of the effected signal to add (0.0 to 1.0). 
        Returns:
            np.ndarray: The processed audio signal.

        Raises:
            ValueError: If the `sampling_rate` is not provided in `kwargs`.

        """
        sampling_rate = kwargs.get("sampling_rate", None)
        start_delays = kwargs.get( "start_delays_chorus", self.config.get("start_delays_chorus"))
        w_delays = kwargs.get("w_delays_chorus",self.config.get("w_delays_chorus"))
        delay_rates = kwargs.get("delay_rates_chorus",self.config.get("delay_rates_chorus"))
        self.dry_gain = kwargs.get("dry_gain_chorus",self.config.get("dry_gain_chorus"))
        self.chorus_gains = kwargs.get("chorus_gains_chorus",self.config.get("chorus_gains_chorus"))

        self.chorus_count = len(self.chorus_gains)


    
        # Multiple chorus
        max_delay = 0
        self.lfo_array = []
        self.chorus_delays = np.zeros(self.chorus_count)

        for i in range(self.chorus_count):
            self.chorus_delays[i] = math.floor(sampling_rate * start_delays[i])
            width = math.floor(sampling_rate * w_delays[i])
            max_delay_i = self.chorus_delays[i] + width + 2
            if max_delay_i > max_delay:
                max_delay = max_delay_i
            self.lfo_array.append(self.LFO(sampling_rate, delay_rates[i], width))   
        
        self.delay_line = self.Delay(int(max_delay))  # one delay line used for all paths


        filtered=np.zeros_like(audio)
        for i in range(len(audio)):
            filtered[i]=self.apply_one_step(audio[i])

        filtered = filtered / max(np.abs(filtered))
        return filtered


    def apply_one_step(self, x: float) -> float:
        y = x * self.dry_gain
        for i in range(self.chorus_count):
            lfo = self.lfo_array[i]
            tap = self.chorus_delays[i] + lfo.tick()
            d = math.floor(tap)

            # Linear Interpolation 
            frac = tap - d
            candidate1 = self.delay_line.go_back(d)
            candidate2 = self.delay_line.go_back(d + 1)
            interp = frac * candidate2 + (1 - frac) * candidate1
            y += interp * self.chorus_gains[i]
        self.delay_line.push(x)
        return y
        

    class LFO:
        def __init__(self, sample_rate: int, frequency: float, width: float, 
                    waveform: str = 'sine', offset: float = 0.0, bias: float = 0.0) -> None:
            """
            Low-Frequency Oscillator (LFO) for modulating parameters such as delay time.
            Args:
                sample_rate (int): Sampling rate of the audio signal.
                frequency (float): Frequency of the LFO in Hz.
                width (float): Amplitude of the modulation 
                waveform (str): Type of waveform ('sine' only currently supported).
                offset (float): Initial phase offset.
                bias (float): DC offset to add to the output signal.
            """
            self.waveform = waveform
            self.width = width
            self.delta = frequency / sample_rate
            self.phase = offset
            self.bias = bias

        def process(self, n: int) -> float:
            """
            Compute the LFO value at sample index `n`.
            Args:
                n (int): The sample index.
            Returns:
                float: The modulated value at index `n`.
            """
            return self.width * math.sin(2 * math.pi * self.delta * n) + self.bias

        def tick(self, i: int = 1) -> float:
            """
            Advances the LFO by `i` samples and returns the current output value.

            Args:
                i (int): Number of steps to advance the LFO. Default is 1.

            Returns:
                float: Current modulated value based on the phase.
            """
            ret = self.width * math.sin(2 * math.pi * self.phase) + self.bias
            self.phase += i * self.delta
            if self.phase > 1.0:
                self.phase -= 1.0
            return ret

    class Delay:
        def __init__(self, delay_length: int) -> None:
            """
            Implements a circular delay line for audio processing.
            Args:
                delay_length (int): Total length of the delay buffer.
            """
            self.length = delay_length - 2  
            self.buffer = np.zeros(delay_length, dtype=np.float32)
            self.pos = 0

        def front(self) -> float:
            """
            Returns the current value at the write position (front of the buffer).
            Returns:
                float: The sample currently at the front of the buffer.
            """
            return self.buffer[self.pos]

        def push(self, x: float) -> None:
            """
            Pushes a new sample into the delay buffer.
            Args:
                x (float): The new audio sample to add.
            """
            self.buffer[self.pos] = x
            self.pos += 1
            if self.pos + 1 >= self.length:
                self.pos -= self.length

        def go_back(self, idx: int) -> float:
            """
            Retrieves a past sample from the delay buffer.
            Args:
                idx (int): How far back to go in the buffer.
            Returns:
                float: The sample from `idx` steps ago.
            """
            target = self.pos - idx
            if target < 0:
                target += self.length
            if (target>=len(self.buffer)):
                target=target%len(self.buffer)
            return self.buffer[target]