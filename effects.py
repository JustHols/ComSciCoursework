# effects.py

# IMPORTS
import numpy as np

# SUBPROGRAMS
def lowpass(data: np.ndarray[np.float64],
            samplerate: int,
            settings: dict[str, float],
            history: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
    """
    Lowpass filter audio processing effect.

    Args:
         data (np.ndarray[np.float64]): The input data to be processed.
         samplerate (int): The samplerate for that data.
         settings (dict[str, float]): The effect's current settings.
         history (np.ndarray[np.float64]): History used for live processing.
    """
    # if live processing on add last value from previous block
    # to start of current block to ensure smooth processing
    if history is not None:
        prev = history[-1]
        data = np.append(prev, data)

    # get current settings for effect
    frequency = settings["freq/Hz"]

    RC = 1.0 / (2 * np.pi * frequency)
    dt = 1.0 / samplerate
    alpha = dt / (RC + dt)

    # create copy of input data just containing 0s to add processed values to
    new_data = np.zeros_like(data)
    new_data[0] = alpha * data[0]

    # recurrence relation
    for i in range(1, len(data)):
        new_data[i] = new_data[i - 1] + alpha * (data[i] - new_data[i - 1])

    # ensure all amplitudes within valid range for np.float64
    new_data = np.clip(new_data, -1.0, 1.0)
    # if live the first value is from previous buffer so don't return it
    if history is not None:
        return new_data[1:]
    return new_data

def highpass(data: np.ndarray[np.float64],
             samplerate: int,
             settings: dict[str, float],
             history: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
    """
    Highpass filter audio processing effect.

    Args:
         data (np.ndarray[np.float64]): The input data to be processed.
         samplerate (int): The samplerate for that data.
         settings (dict[str, float]): The effect's current settings.
         history (np.ndarray[np.float64]): History used for live processing.
    """
    # if live processing on add last value from previous block
    # to start of current block to ensure smooth processing
    if history is not None:
        prev = history[-1]
        data = np.append(prev, data)

    # get current settings for effect
    frequency = settings["freq/Hz"]

    RC = 1 / (2 * np.pi * frequency)
    dt = 1 / samplerate
    alpha = RC / (RC + dt)

    # create copy of input data just containing 0s to add processed values to
    new_data = np.zeros_like(data)
    new_data[0] = data[0]

    # recurrence relation
    for i in range(1, len(data)):
        new_data[i] = alpha * (new_data[i - 1] + data[i] - data[i - 1])

    # ensure all amplitudes within valid range for np.float64
    new_data = np.clip(new_data, -1.0, 1.0)
    # if live the first value is from previous buffer so don't return it
    if history is not None:
        return new_data[1:]
    return new_data

def bitcrusher(data: np.ndarray[np.float64],
               samplerate: int,
               settings: dict[str, float],
               history: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
    """
    Bitcrusher audio processing effect.

    Args:
         data (np.ndarray[np.float64]): The input data to be processed.
         samplerate (int): The samplerate for that data.
         settings (dict[str, float]): The effect's current settings.
         history (np.ndarray[np.float64]): History used for live processing.
    """
    # get current settings for effect
    new_samplerate = settings["samplerate/Hz"]
    bit_depth = settings["bits/sample"]

    # changing samplerate
    ratio = max(1, samplerate // new_samplerate)
    new_data = data[(np.arange(len(data)) // ratio) * ratio]

    # changing bit_depth
    scale = 2 ** (bit_depth - 1)
    new_data = np.round(new_data * scale) / scale

    # ensure all amplitudes within valid range for np.float64
    new_data = np.clip(new_data, -1.0, 1.0)
    return new_data

def volume(data: np.ndarray[np.float64],
           samplerate: int,
           settings: dict[str, float],
           history: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
    """
    Volume changing audio processing effect.

    Args:
         data (np.ndarray[np.float64]): The input data to be processed.
         samplerate (int): The samplerate for that data.
         settings (dict[str, float]): The effect's current settings.
         history (np.ndarray[np.float64]): History used for live processing.
    """
    # get current settings for effect
    gain = settings["gain/dB"]

    # convert dB to amplitude scale factor
    gain = 10 ** (gain / 20)
    new_data = data * gain

    # ensure all amplitudes within valid range for np.float64
    new_data = np.clip(new_data, -1.0, 1.0)
    return new_data

def pitch_shifter(data: np.ndarray[np.float64],
                  samplerate: int,
                  settings: dict[str, float],
                  history: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
    """
    Pitch changer audio processing effect.

    Args:
         data (np.ndarray[np.float64]): The input data to be processed.
         samplerate (int): The samplerate for that data.
         settings (dict[str, float]): The effect's current settings.
         history (np.ndarray[np.float64]): History used for live processing.
    """
    # get current settings for effect
    semitones = settings["semitone shift"]

    # find ratio representing frequency scale factor
    ratio = 2 ** (semitones / 12.0)
    # find new length of sample required to change pitch
    length = int(len(data) / ratio)

    new_data = np.interp(np.linspace(0, len(data) - 1, length),
                         np.arange(len(data)), data)

    # ensure all amplitudes within valid range for np.float64
    new_data = np.clip(new_data, -1.0, 1.0)
    return new_data

def delay(data: np.ndarray[np.float64],
          samplerate: int,
          settings: dict[str, float],
          history: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
    """
    Delay audio processing effect.

    Args:
         data (np.ndarray[np.float64]): The input data to be processed.
         samplerate (int): The samplerate for that data.
         settings (dict[str, float]): The effect's current settings.
         history (np.ndarray[np.float64]): History used for live processing.
    """
    # get current settings for effect
    delay = settings["delay/s"]
    decay = settings["decay"]

    # calculates index of input data when delayed track will start
    delay_index = int(samplerate * delay)

    # if live processing on add relevant history to start of
    # input data and make new np.ndarray to store processed data
    if history is not None:
        history = history[-delay_index:]
        new_data = np.append(history, data)
    # make new np.ndarray to store processed data when not live
    else:
        new_data = np.append(data, np.zeros(delay_index))

    # adds delayed track to original data
    for i in range(delay_index, len(new_data)):
        new_data[i] += (1 - decay) * new_data[i - delay_index]

    # ensure all amplitudes within valid range for np.float64
    new_data = np.clip(new_data, -1.0, 1.0)
    # for real time don't return previous buffers
    if history is not None:
        return new_data[-len(data):]
    return new_data

def flanger(data: np.ndarray[np.float64],
            samplerate: int,
            settings: dict[str, float],
            history: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
    """
    Flanger audio processing effect.

    Args:
         data (np.ndarray[np.float64]): The input data to be processed.
         samplerate (int): The samplerate for that data.
         settings (dict[str, float]): The effect's current settings.
         history (np.ndarray[np.float64]): History used for live processing.
    """
    # get current settings for effect
    rate = settings["rate/Hz"]
    max_delay = settings["max delay/s"]
    mix = settings["mix"]

    # calculate the maximum delay in samples
    max_delay_index = int(max_delay * samplerate)

    # add relevant history if in live mode to start of input data
    if history is not None:
        history = history[-max_delay_index:]
        data = np.append(history, data)

    # make np.ndarray for processed data
    new_data = np.copy(data)

    # low frequency oscillator
    lfo = np.sin(2 * np.pi * rate / samplerate * np.arange(len(new_data)))
    # modulated delays
    delays = ((lfo + 1) / 2 * max_delay_index).astype(int)

    # mix input data and the data processed with the modulated delay together
    for i in range(len(new_data)):
        if i - delays[i] >= 0:
            delayed = data[i - delays[i]]
            new_data[i] += mix * delayed

    # ensure all amplitudes within valid range for np.float64
    new_data = np.clip(new_data, -1.0, 1.0)
    # don't return previous buffers again
    if history is not None:
        return new_data[-len(data):]
    return new_data