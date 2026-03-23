# model.py

# IMPORTS
import numpy as np
import soundfile as sf
import sounddevice as sd
import json
import effects

# CLASSES
class Model:
    """
    Class representing the model part of the model, view,
    controller structure. Describes what different effects are
    and how to manipulate the audio data.

    Attributes:
        __effects_info (dict[str, dict]): Basic information
                                          about each effect.
        __live (bool): Flag to show whether
                       live audio processing active.

    Methods:
        add_effect(data: np.ndarray[np.float64],
                   samplerate: int,
                   name: str,
                   settings: dict[str, float],
                   history: np.ndarray[np.float64]) -> np.ndarray[np.float64]
        get_effects_info() -> dict[str, dict]
        stop_audio() -> None
        process_chain(chain: dict[str, dict[str, float]],
                      data: np.ndarray[np.float64],
                      samplerate: int,
                      history: np.ndarray[np.float64]) -> np.ndarray[np.float64]
        start_live_processing(chain_callback) -> None
        stereo_to_mono(data: np.ndarray[np.float64]) -> np.ndarray[np.float64]
        play_audio_data(data: np.ndarray[np.float64]) -> None
        read_audio_file(filepath: str) -> tuple[np.ndarray[np.float64], int]
        save_file(filepath: str,
                  data: np.ndarray[np.float64],
                  samplerate: int) -> None
        save_preset(filepath: str,
                    chain: dict[str, dict[str, float]]) -> None
        load_preset(filepath: str) -> dict[str, dict[str, float]]
    """
    def __init__(self) -> None:

        # ATTRIBUTES
        self.__effects_info = {
            "LOWPASS": {
                "function": effects.lowpass,
                "colour": "#F94144",
                "setup": {
                    "freq/Hz": [20, 20000, 20, 20000],
                },
            },
            "HIGHPASS": {
                "function": effects.highpass,
                "colour": "#F3722C",
                "setup": {
                    "freq/Hz": [20, 20000, 20, 20],
                },
            },
            "BITCRUSHER": {
                "function": effects.bitcrusher,
                "colour": "#F8961E",
                "setup": {
                    "samplerate/Hz": [1000, 44100, 100, 44100],
                    "bits/sample": [4, 64, 1, 64],
                },
            },
            "VOLUME": {
                "function": effects.volume,
                "colour": "#f9C74F",
                "setup": {
                    "gain/dB": [-20, 20, 1, 0],
                },
            },
            "PITCH SHIFTER": {
                "function": effects.pitch_shifter,
                "colour": "#90BE6D",
                "setup": {
                    "semitone shift": [-12, 12, 1, 0],
                },
            },
            "DELAY": {
                "function": effects.delay,
                "colour": "#43AA8B",
                "setup": {
                    "delay/s": [0, 1.5, 0.01, 0],
                    "decay": [0, 1, 0.1, 1],
                },
            },
            "FLANGER": {
                "function": effects.flanger,
                "colour": "#577590",
                "setup": {
                    "rate/Hz": [0, 5, 0.1, 0],
                    "mix": [0, 0.8, 0.1, 0],
                    "max delay/s": [0, 0.1, 0.01, 0],
                },
            },
        }
        self.__live = False

    # METHODS
    def stereo_to_mono(self,
                       data: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
        """
        If the inputted audio data has more than 1
        channel converts it to only have 1 channel.

        Args:
            data (np.ndarray[np.float64]): The audio data to be checked
                                           and possibly converted.

        Returns:
            np.ndarray[np.float64]: The mono audio data.
        """
        if data.ndim == 1:
            return data
        return data.mean(axis=1)

    def play_audio_data(self,
                        data: np.ndarray[np.float64],
                        samplerate: int) -> None:
        """
        Plays inputted audio data at the given samplerate.

        Args:
            data (np.ndarray[np.float64]): The audio data to be played.
            samplerate (int): The samplerate for that data.
        """
        sd.play(data, samplerate)
        sd.wait()

    def read_audio_file(self,
                        filepath: str) -> tuple[np.ndarray[np.float64], int]:
        """
        Reads the data and samplerate from a .wav file with a specified filepath.

        Args:
            filepath (str): Path to the .wav file.

        Returns:
            tuple[np.ndarray[np.float64], int]: A tuple containing the data
                                                and samplerate of the file.
        """
        data, samplerate = sf.read(filepath)
        data = self.stereo_to_mono(data)
        return data, samplerate

    def save_file(self,
                  filepath: str, #
                  data: np.ndarray[np.float64],
                  samplerate: int) -> None:
        """
        Saves the given audio data at the specified samplerate
        at location given by filepath as .wav file.

        Args:
            filepath (str): Location to save the .wav file.
            data (np.ndarray[np.float64]): The audio data to be saved.
            samplerate (int): The samplerate for that data.
        """
        sf.write(filepath, data, samplerate)

    def save_preset(self,
                    filepath: str,
                    chain: dict[str, dict[str, float]]) -> None:
        """
        Saves the effects chain dictionary as a .json
        file in the location specified by filepath.

        Args:
            filepath (str): Location to save the .json file.
            chain (dict[str, dict[str, float]]): Effects chain dictionary of
                                                 currently active effects and
                                                 their settings.
        """
        with open(filepath, "w") as file:
            json.dump(chain, file)

    def load_preset(self,
                    filepath: str) -> dict[str, dict[str, float]]:
        """
        Reads .json file located at filepath returning
        a dictionary representing the effects chain.

        Args:
            filepath (str): Path to the .json file.

        Returns:
            dict[str, dict[str, float]]: The effects chain dictionary
                                         for the preset being loaded.
        """
        with open(filepath, "r") as file:
            return json.load(file)

    def add_effect(self,
                   data: np.ndarray[np.float64],
                   samplerate: int,
                   name: str,
                   settings: dict[str, float],
                   history: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
        """
        Processes inputted audio data to add effect specified by name and settings.

        Args:
            data (np.ndarray[np.float64]): Input audio data.
            samplerate (int): Samplerate of that data.
            name (str): Name of the effect to be added.
            settings (dict[str, float]): Current settings of the effect.
            history (np.ndarray[np.float64]): Array of previous blocks
                                              used for live processing.

        Returns:
            np.ndarray[np.float64]: The processed audio data.
        """
        return self.__effects_info[name]["function"](data, samplerate, settings, history)

    def get_effects_info(self) -> dict[str, dict]:
        """
        Getter for __effects_info:

        Returns:
            dict[str, dict]: The information on the effects.
        """
        return self.__effects_info

    def stop_audio(self) -> None:
        """
        When called stops any live audio processing
        as well as any other playing audio in any thread.
        """
        self.__live = False
        sd.stop()

    def process_chain(self,
                      chain: dict[str, dict[str, float]],
                      data: np.ndarray[np.float64],
                      samplerate: int,
                      history: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
        """
        Adds each effect in the chain of effects
        to input audio data and returns processed result.

        Args:
            chain (dict[str, dict[str, float]]): Chain of effects.
            data (np.ndarray[np.float64]): Input audio data.
            samplerate (int): Samplerate of that data.
            history (np.ndarray[np.float64]): Array of previous blocks
                                              used for live processing.

        Returns:
            np.ndarray[np.float64]: The processed audio data.
        """
        for name, settings in chain.items():
            # if not live processing just continue but
            # if live pitch shifter isn't available so ignore
            if (self.__live and name != "PITCH SHIFTER") or not self.__live:
                data = self.add_effect(data, samplerate, name, settings, history)
        return data

    def start_live_processing(self, chain_callback) -> None:
        """
        Begins live processing by creating an audio stream to read
        input audio data from which will be processed before
        being written back to the stream. Previous blocks are
        recorded as history in order for some effects to work live.

        Args:
            chain_callback: Function that will be called to get
                            updated chain each time a block processed.
        """
        self.__live = True

        # LOCAL VARIABLES
        samplerate = 44100
        blocksize = 1024
        history = np.array([])
        # setting up stream object
        stream = sd.Stream(samplerate=samplerate,
                           blocksize=blocksize,
                           channels=1,
                           dtype=np.float32)
        # calculating maximum length of history required
        # to store from samplerate * maximum delay time
        max_len_history = int(samplerate * self.__effects_info["DELAY"]["setup"]["delay/s"][1])

        # starting stream safely
        with stream:
            # keep looping whilst live mode is active
            while self.__live:
                # update chain
                chain = chain_callback()

                # read, process and then write new data
                data = stream.read(blocksize)[0].astype(np.float64)
                new_data = self.process_chain(chain, data, samplerate, history)
                stream.write(new_data.astype(np.float32))

                # add last block to history and cut history if too large
                history = np.append(history, data)
                if len(history) > max_len_history:
                    history = history[-max_len_history:]