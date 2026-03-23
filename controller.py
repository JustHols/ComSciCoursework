# controller.py

# IMPORTS
import model
import numpy as np

# CLASSES
class Controller:
    """
    Class representing the controller part of the model,
    view, controller structure. Mediates the
    flow of data between the model and the view.

    Attributes:
        __model (Model): A model object.
                         Part of the MVC system.

    Methods:
        get_effects_info() -> dict[str: dict]
        play_file(filepath: str) -> None
        play_data(filepath: str) -> None
        stop_audio() -> None
        save_file(filepath: str,
                  data: np.ndarray[np.float64],
                  samplerate: int) -> None
        save_preset(filepath: str,
                    chain: dict[str, dict[str, float]]) -> None
        load_preset(filepath: str) -> dict[str, dict[str, float]]
        process_chain(chain: dict[str, dict[str, float]],
                      filepath: str) -> tuple[np.ndarray[np.float64], int]
        start_live_processing(chain_callback) -> None
    """
    def __init__(self) -> None:
        # ATTRIBUTES
        self.__model = model.Model()

    # METHODS
    def get_effects_info(self) -> dict[str: dict]:
        """
        Calls the model's getter for the __effects_info.

        Returns:
            dict[str, dict]: Dictionary of effects information.
        """
        return self.__model.get_effects_info()

    def play_file(self, filepath: str) -> None:
        """
        Asks the model to read the data and samplerate from
        the audio file at specified path before asking the
        model to play the audio data.

        Args:
            filepath (str): Path of the .wav file to be played.
        """
        data, samplerate = self.__model.read_audio_file(filepath)
        self.__model.play_audio_data(data, samplerate)

    def play_data(self,
                  data: np.ndarray[np.float64],
                  samplerate: int) -> None:
        """
        Asks the model to play the audio data
        at the specified samplerate.

        Args:
            data (np.ndarray[np.float64]): Data to be played.
            samplerate (int): Sampling rate of that data.
        """
        self.__model.play_audio_data(data, samplerate)

    def stop_audio(self) -> None:
        """
        Stops any currently playing audio including
        live audio streams.
        """
        self.__model.stop_audio()

    def save_file(self,
                  filepath: str,
                  data: np.ndarray[np.float64],
                  samplerate: int) -> None:
        """
        Saves a .wav file at the specified location
        with given data and samplerate.

        Args:
            filepath (str): Path of the .wav file to be saved.
            data (np.ndarray[np.float64]): Data to be saved.
            samplerate (int): Sampling rate of the data.
        """
        self.__model.save_file(filepath, data, samplerate)

    def save_preset(self,
                    filepath: str,
                    chain: dict[str, dict[str, float]]) -> None:
        """
        Asks model to save a preset and the specified
        location built from the effects chain.

        Args:
            filepath (str): Path of the .json file to be saved.
            chain (dict[str, dict[str, float]]): Effects chain dictionary.
        """
        self.__model.save_preset(filepath, chain)

    def load_preset(self,
                    filepath: str) -> dict[str, dict[str, float]]:
        """
        Asks the model to load a preset from the specified
        location and to convert the .json file into a
        dictionary representing the effects chain.

        Args:
            filepath (str): Path of the .json file to be loaded.

        Returns:
            dict[str, dict[str, float]]: Dictionary representing
                                         the effects chain.
        """
        return self.__model.load_preset(filepath)

    def process_chain(self,
                      chain: dict[str, dict[str, float]],
                      filepath: str) -> tuple[np.ndarray[np.float64], int]:
        """
        Asks model to read the data and samplerate of a .wav file
        stored at the specified filepath before returning the
        data in the file but with all effects in chain applied.

        Args:
            chain (dict[str, dict[str, float]]): Effects chain dictionary.
            filepath (str): Path of the .wav file to be read and processed.

        Returns:
            np.ndarray[np.float64]: Processed data.
        """
        data, samplerate = self.__model.read_audio_file(filepath)
        return self.__model.process_chain(chain, data, samplerate, None), samplerate

    def start_live_processing(self, chain_callback) -> None:
        """
        Asks model to start a stream for live audio processing using
        the chain_callback function to constantly update the
        current state of the effects chain.

        Args:
            chain_callback: Callback function so model
                            can get current chain state from view.
        """
        self.__model.start_live_processing(chain_callback)