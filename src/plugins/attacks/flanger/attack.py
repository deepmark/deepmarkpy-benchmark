import numpy as np
import math

from core.base_attack import BaseAttack

class FlangerAttack(BaseAttack):

    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        """
        This attack performs a flanger attack on an audio signal, that's been included in the StirMark Benchmark paper.
        Implementation of this attack is from here: https://github.com/chenwj1989/pafx.
        Args:
            audio (np.ndarray): The input audio signal.
            **kwargs: Additional parameters for the flanger attack:
                - sampling_rate (int): The sampling rate of the audio signal in Hz (required).
                - start_delay_flanger (float): Starting amount of time (in seconds) that the input signal is delayed before modulation.
                - w_delay_flanger (float): Controls how much the delay time is varied by the sine wave (amplitude).
                - delay_rate_flanger (float): Frequency (Hz) of the modulation sine wave — typically a low frequency (e.g., 0.3-5 Hz).
                - gain_flanger (float): Strength of the flanger effect.
        Returns:
            np.ndarray: The processed audio signal.

        Raises:
            ValueError: If the `sampling_rate` is not provided in `kwargs`.

        """
        sampling_rate = kwargs.get("sampling_rate", None)
        start_delay = kwargs.get( "start_delay_flanger", self.config.get("start_delay_flanger"))
        w_delay = kwargs.get("w_delay_flanger",self.config.get("w_delay_flanger"))
        delay_rate = kwargs.get("delay_rate_flanger",self.config.get("delay_rate_flanger"))
        self.gain = kwargs.get("gain_flanger",self.config.get("gain_flanger"))

        self.avg_delay = math.floor(sampling_rate * start_delay)
        width = math.floor(sampling_rate * w_delay)
        max_delay = self.avg_delay + width + 2

        self.delay_line = self.Delay(max_delay)
        self.lfo = self.LFO(sampling_rate, delay_rate, width)

        filtered=np.zeros_like(audio)
        for i in range(len(audio)):
            filtered[i] = self.apply_one_step(audio[i])
        
        return filtered


    def apply_one_step(self, x:float) -> float:
        tap = self.avg_delay + self.lfo.tick()
        i = math.floor(tap)

        # Linear Interpolation 
        frac = tap - i
        candidate1 = self.delay_line.go_back(i)
        candidate2 = self.delay_line.go_back(i + 1)
        interp = frac * candidate2 + (1 - frac) * candidate1

        self.delay_line.push(x)
        return interp * self.gain + x
    

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
            Implements a circular delay buffer for audio processing.
            Args:
                delay_length (int): Total length of the delay buffer.
            """
            self.length: int = delay_length
            self.buffer: np.ndarray = np.zeros(delay_length)
            self.pos: int = 0

        def front(self) -> float:
            """
            Returns the current value at the write position.
            Returns:
                float: Sample at the current write position.
            """
            return self.buffer[self.pos]

        def push(self, x: float) -> None:
            """
            Inserts a new sample into the delay buffer.
            Args:
                x (float): New sample to insert.
            """
            self.buffer[self.pos] = x
            self.pos += 1
            if self.pos +1 >= self.length:
                self.pos -= self.length

        def go_back(self, idx: int) -> float:
            """
            Retrieves a sample from `idx` steps ago in the delay buffer.
            Args:
                idx (int): How many samples back to access.
            Returns:
                float: The delayed sample.
            """
            target = self.pos - idx
            if target < 0 :
                target = target + self.length 
            return self.buffer[target]