import numpy as np
from scipy.signal import butter, filtfilt
import UnicornPy
import pylsl
import time

# Define constants for processing
LOWCUT = 75
HIGHCUT = 120
SAMPLE_RATE = 250  # Unicorn sample rate in Hz
STREAM_RATE = 90  # Vision Pro max rate in Hz
FRAME_LENGTH = 1  # Length of each frame in seconds
BUFFER_SIZE = FRAME_LENGTH * SAMPLE_RATE  # Buffer size in samples
ACQUIRED_CHANNELS = 8  # Number of channels we are interested in (first 8 channels)
PROCESSING_WINDOW = 1  # Processing window in seconds (must be greater than padlen/SAMPLE_RATE)
PADLEN = 27  # Padding length required by filtfilt

# Define LSL stream info and outlet
stream_info = pylsl.StreamInfo('UnicornStream', 'EEG', 4, STREAM_RATE, 'float32', 'unicorn12345')
outlet = pylsl.StreamOutlet(stream_info)

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data, padlen=PADLEN)
    return y

def process_data(raw_data):
    # Use only the first 8 channels for bipolar configuration
    raw_data = raw_data[:, :ACQUIRED_CHANNELS]
    
    # Implement bipolar configuration for each pair of electrodes
    bipolar_data = np.zeros((raw_data.shape[0], 4))
    bipolar_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
    
    for i, (ch1, ch2) in enumerate(bipolar_pairs):
        bipolar_data[:, i] = raw_data[:, ch1] - raw_data[:, ch2]

    # Process the data
    processed_data = np.zeros((bipolar_data.shape[0], 4))
    
    for i in range(4):
        filtered_signal = bandpass_filter(bipolar_data[:, i], LOWCUT, HIGHCUT, SAMPLE_RATE)
        rectified_signal = np.abs(filtered_signal)
        mean_power_signal = np.mean(rectified_signal)
        processed_data[:, i] = mean_power_signal

    return processed_data

def downsample_data(data, original_rate, target_rate):
    factor = int(original_rate / target_rate)
    return data[::factor]

def main():
    # Specifications for the data acquisition
    testsignale_enabled = False

    print("Unicorn Acquisition Example")
    print("---------------------------")
    print()

    try:
        # Get available devices
        device_list = UnicornPy.GetAvailableDevices(True)

        if len(device_list) <= 0 or device_list is None:
            raise Exception("No device available. Please pair with a Unicorn first.")

        # Print available device serials
        print("Available devices:")
        for i, device in enumerate(device_list):
            print(f"#{i} {device}")

        # Select the first device for simplicity
        device_id = 0
        if device_id < 0 or device_id >= len(device_list):
            raise IndexError('The selected device ID is not valid.')

        # Open selected device
        print(f"Trying to connect to '{device_list[device_id]}'.")
        device = UnicornPy.Unicorn(device_list[device_id])
        print(f"Connected to '{device_list[device_id]}'.")
        print()

        # Initialize acquisition members
        number_of_acquired_channels = device.GetNumberOfAcquiredChannels()
        print(f"Number of acquired channels: {number_of_acquired_channels}")
        print(f"Sampling Rate: {UnicornPy.SamplingRate} Hz")
        print(f"Frame Length: {FRAME_LENGTH} seconds")

        # Allocate memory for the acquisition buffer
        receive_buffer_length = BUFFER_SIZE * number_of_acquired_channels * 4
        receive_buffer = bytearray(receive_buffer_length)

        # Initialize a buffer to accumulate data
        accumulated_data = np.zeros((0, number_of_acquired_channels))

        try:
            # Start data acquisition
            device.StartAcquisition(testsignale_enabled)
            print("Data acquisition started.")

            while True:
                # Receive the configured number of samples from the Unicorn device
                device.GetData(BUFFER_SIZE, receive_buffer, receive_buffer_length)

                # Convert receive buffer to numpy float array
                raw_data = np.frombuffer(receive_buffer, dtype=np.float32, count=number_of_acquired_channels * BUFFER_SIZE)
                raw_data = np.reshape(raw_data, (BUFFER_SIZE, number_of_acquired_channels))

                # Accumulate data
                accumulated_data = np.vstack((accumulated_data, raw_data))

                # If we have enough data for processing
                if accumulated_data.shape[0] >= PROCESSING_WINDOW * SAMPLE_RATE:
                    # Process data
                    processed_data = process_data(accumulated_data)

                    # Downsample the processed data to 90 Hz
                    downsampled_data = downsample_data(processed_data, SAMPLE_RATE, STREAM_RATE)

                    # Stream data
                    for sample in downsampled_data:
                        outlet.push_sample(sample)
                        print(f"Streamed sample: {sample}")  # Print streamed data

                    # Clear the accumulated data
                    accumulated_data = np.zeros((0, number_of_acquired_channels))

        except UnicornPy.DeviceException as e:
            print(e)
        except Exception as e:
            print(f"An unknown error occurred: {e}")
        finally:
            # Close device
            del device
            print("Disconnected from Unicorn")

    except UnicornPy.DeviceException as e:
        print(e)
    except Exception as e:
        print(f"An unknown error occurred: {e}")

if __name__ == "__main__":
    main()
