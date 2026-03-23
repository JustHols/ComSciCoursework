import pytest
import numpy as np
import effects as e

lowpass_test_data = [
    (
        np.array([0.0, 0.524, 1.0, -1.0, -0.63], dtype=np.float64),
        44100,
        {"freq/Hz":300},
        np.array([0.0, 0.02147913, 0.06158938, 0.01807409, -0.00849092], dtype=np.float64)
    ),
    (
        np.array([0.5, 0.1, -0.1], dtype=np.float64),
        44100,
        {"freq/Hz":200},
        np.array([0.01385285, 0.01623961, 0.01301911], dtype=np.float64)
    )
]

highpass_test_data = [
    (
        np.array([0.0, 0.524, 1.0, -1.0, -0.63], dtype=np.float64),
        44100,
        {"freq/Hz":300},
        np.array([0.0, 0.50252087, 0.93841062, -1.0, -0.62150908], dtype=np.float64)
    ),
    (
        np.array([0.5, 0.1, -0.1], dtype=np.float64),
        44100,
        {"freq/Hz": 200},
        np.array([0.5, 0.09722943, -0.09992324], dtype=np.float64)
    )
]

flanger_test_data = [
    (
        np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64),
        44100,
        {"rate/Hz": 0.5, "max delay/s": 0.001, "mix": 0.5}
    ),
    (
        np.array([0.0, 0.0, 0.0], dtype=np.float64),
        44100,
        {"rate/Hz": 1.0, "max delay/s": 0.002, "mix": 1.0}
    )
]

bitcrusher_test_data = [
    (
        np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64),
        44100,
        {"samplerate/Hz": 22050, "bits/sample": 2},
        np.array([0.0, 0.0, 0.5, 0.5, 1.0], dtype=np.float64)
    ),
    (
        np.array([0.1, -0.1, 0.9], dtype=np.float64),
        44100,
        {"samplerate/Hz": 44100, "bits/sample": 8},
        np.array([0.1015625, -0.1015625, 0.8984375], dtype=np.float64)
    )
]

delay_test_data = [
    (
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        1,
        {"delay/s": 1, "decay": 0.5},
        np.array([1.0, 0.5, 0.25, 0.125], dtype=np.float64)
    ),
]

volume_test_data = [
    (
        np.array([0.1, -0.1, 0.5], dtype=np.float64),
        44100,
        {"gain/dB": 6},
        np.clip(np.array([0.1, -0.1, 0.5]) * (10 ** (6 / 20)), -1.0, 1.0)
    ),
    (
        np.array([0.8, -0.8], dtype=np.float64),
        44100,
        {"gain/dB": 20},
        np.array([1.0, -1.0], dtype=np.float64)
    )
]

pitch_test_data = [
    (
        np.array([0.0, 0.5, 1.0, 0.5, 0.0], dtype=np.float64),
        44100,
        {"semitone shift": 12},
        2
    ),
    (
        np.array([0.1, 0.2, 0.3], dtype=np.float64),
        44100,
        {"semitone shift": 0},
        3
    )
]

@pytest.mark.parametrize("input_data, samplerate, settings, output_data", lowpass_test_data)
def test_lowpass(input_data, samplerate, settings, output_data):
    assert np.allclose(e.lowpass(input_data, samplerate, settings, None), output_data, atol=1e-6)

@pytest.mark.parametrize("input_data, samplerate, settings, output_data", highpass_test_data)
def test_highpass(input_data, samplerate, settings, output_data):
    assert np.allclose(e.highpass(input_data, samplerate, settings, None), output_data, atol=1e-6)

@pytest.mark.parametrize("input_data, samplerate, settings, output_data", bitcrusher_test_data)
def test_bitcrusher(input_data, samplerate, settings, output_data):
    assert np.allclose(e.bitcrusher(input_data, samplerate, settings, None), output_data)

@pytest.mark.parametrize("input_data, samplerate, settings, output_data", volume_test_data)
def test_volume(input_data, samplerate, settings, output_data):
    assert np.allclose(e.volume(input_data, samplerate, settings, None), output_data)

@pytest.mark.parametrize("input_data, samplerate, settings, expected_length", pitch_test_data)
def test_pitch_shifter(input_data, samplerate, settings, expected_length):
    result = e.pitch_shifter(input_data, samplerate, settings, None)
    assert len(result) == expected_length

@pytest.mark.parametrize("input_data, samplerate, settings, output_data", delay_test_data)
def test_delay(input_data, samplerate, settings, output_data):
    result = e.delay(input_data, samplerate, settings, None)
    assert np.allclose(result[:len(output_data)], output_data)

@pytest.mark.parametrize("input_data, samplerate, settings", flanger_test_data)
def test_flanger(input_data, samplerate, settings):
    result = e.flanger(input_data, samplerate, settings, None)
    assert len(result) == len(input_data) and np.all((-1.0 <= result) & (result <= 1.0))