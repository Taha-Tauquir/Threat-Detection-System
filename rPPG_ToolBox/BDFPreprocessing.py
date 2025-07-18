import mne
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.signal import find_peaks
from biosppy.signals import ecg as biosppy_ecg

class BDFSyncProcessor:
    def __init__(self, bdf_path):
        self.bdf_path = bdf_path
        self.raw = mne.io.read_raw_bdf(bdf_path, preload=True)
        self.trigger_channel = None
        self.trigger_time = None
        self.trigger_value = None



    def detect_r_peaks_biosppy(self,ecg_data, fs):
        # biosppy returns a dictionary with R-peak locations in samples
        output = biosppy_ecg.ecg(signal=ecg_data, sampling_rate=fs, show=False)
        rpeaks = output['rpeaks']
        return rpeaks

    def print_channels(self):
        print("\nAll Channel Names:")
        for idx, name in enumerate(self.raw.ch_names):
            print(f"{idx + 1}: {name}")

    def find_trigger_channel(self):
        keywords = ['trigger', 'status', 'stim']
        trigger_candidates = [ch for ch in self.raw.ch_names if any(k in ch.lower() for k in keywords)]

        if not trigger_candidates:
            print(" No trigger channel found automatically. Please check manually.")
            return None

        self.trigger_channel = trigger_candidates[0]
        print(f" Trigger channel detected: {self.trigger_channel}")
        return self.trigger_channel

    def extract_trigger_signal(self, save_csv=False):
        if self.trigger_channel is None:
            raise RuntimeError("Trigger channel not set. Run find_trigger_channel() first.")

        trigger_data, times = self.raw.get_data(picks=self.trigger_channel, return_times=True)
        trigger_data = trigger_data[0]

        if save_csv:
            df = pd.DataFrame({'Time (s)': times, 'Trigger': trigger_data})
            out_path = os.path.splitext(os.path.basename(self.bdf_path))[0] + "_trigger_signal.csv"
            df.to_csv(out_path, index=False)
            print(f" Trigger signal saved to {out_path}")

        return trigger_data, times

    def detect_first_trigger_time(self, threshold=0.0):
        trigger_data, times = self.extract_trigger_signal()
        indices = np.where(trigger_data > threshold)[0]

        if indices.size == 0:
            print(" No trigger detected in signal.")
            return None

        idx = indices[0]
        self.trigger_time = times[idx]
        self.trigger_value = trigger_data[idx]
        print(f" First trigger detected at {self.trigger_time:.3f}s (value = {self.trigger_value})")
        return self.trigger_time

    def extract_middle_ecg_segment(self, full_duration_sec=135.0, segment_length_sec=30.0, save_csv=True):
        """
        Extracts the middle portion (default 30s) of the full-duration ECG (EXG2) signal.
        Also prints total ECG duration and extracted segment duration.
        Scales ECG from volts to microvolts for better R-peak detection.
        """
        if self.trigger_time is None:
            raise ValueError("Trigger time not set. Run detect_first_trigger_time() first.")

        fs = self.raw.info['sfreq']
        total_samples = self.raw.n_times
        total_duration = total_samples / fs
        print(f" Total ECG Duration from file: {total_duration:.2f} seconds")

        # Calculate middle segment time range
        mid_offset = full_duration_sec / 2
        half_segment = segment_length_sec / 2
        t_start = self.trigger_time + mid_offset - half_segment
        t_end = self.trigger_time + mid_offset + half_segment

        # Crop and extract ECG data
        cropped_raw = self.raw.copy().pick(['EXG2']).crop(tmin=t_start, tmax=t_end)
        ecg_data, times = cropped_raw.get_data(return_times=True)
        ecg_data = ecg_data[0]

        # Convert from volts to microvolts
        ecg_data = ecg_data * 1e6

        # Segment duration check
        segment_duration = times[-1] - times[0]
        print(f" Extracted Middle ECG Segment Duration: {segment_duration:.2f} seconds")

        # Save to CSV if requested
        if save_csv:
            df = pd.DataFrame({'Time (s)': times, 'EXG2 (µV)': ecg_data})
            out_name = os.path.splitext(os.path.basename(self.bdf_path))[
                           0] + f"_middle_{int(segment_length_sec)}s_ecg.csv"
            df.to_csv(out_name, index=False)
            print(f" Middle {segment_length_sec}s ECG saved to: {out_name}")

        return times, ecg_data

    def crop_synced_segment(self, duration=133.0, channels=None, save_csv=False):
        if self.trigger_time is None:
            raise RuntimeError("Trigger time not set. Run detect_first_trigger_time() first.")

        end_time = self.trigger_time + duration
        cropped_raw = self.raw.copy()

        if channels:
            cropped_raw.pick(channels)

        cropped_raw.crop(tmin=self.trigger_time, tmax=end_time)
        data, times = cropped_raw.get_data(return_times=True)

        # Save CSV if needed
        # if save_csv:
        #     df = pd.DataFrame({'Time (s)': times})
        #     for i, ch in enumerate(cropped_raw.ch_names):
        #         df[ch] = data[i]
        #     out_name = os.path.splitext(os.path.basename(self.bdf_path))[0] + "_synced_data.csv"
        #     df.to_csv(out_name, index=False)
        #     print(f" Synced data saved to {out_name}")

        return cropped_raw

    def plot_ecg_signal(self, times, ecg_data, title="Extracted ECG (EXG2)"):
        """
        Plots the ECG signal (EXG2) over time.
        """
        plt.figure(figsize=(12, 4))
        plt.plot(times, ecg_data, linewidth=1.0, color='blue')
        plt.xlabel("Time (s)")
        plt.ylabel("ECG Amplitude (EXG2)")
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def preprocess_ecg(self, ecg_data, fs=256, lowcut=0.5, highcut=40.0):
        """
        Improved ECG preprocessing:
        - Bandpass filter to remove noise and drift
        - Median baseline correction
        - Z-score normalization
        """
        from scipy.signal import medfilt

        # Step 1: Bandpass filter
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(N=4, Wn=[low, high], btype='band')
        filtered = filtfilt(b, a, ecg_data)

        # Step 2: Baseline correction (optional but useful)
        baseline = medfilt(filtered, kernel_size=int(fs * 0.2) | 1)  # odd window size
        corrected = filtered - baseline

        # Step 3: Normalize (optional but helpful for peak detection)
        normalized = (corrected - np.mean(corrected)) / np.std(corrected)

        return normalized

    def estimate_heart_rate(self, times, ecg_data, plot_peaks=True):
        fs = int(1 / (times[1] - times[0]))

        try:
            from biosppy.signals import ecg as biosppy_ecg
        except ImportError:
            raise ImportError("Please install BioSPPy with `pip install biosppy`")

        rpeaks = self.detect_r_peaks_biosppy(ecg_data, fs)
        rpeak_times = times[rpeaks]

        rr_intervals = np.diff(rpeak_times)
        heart_rates = 60 / rr_intervals if len(rr_intervals) > 0 else []
        avg_hr = np.mean(heart_rates) if len(heart_rates) > 0 else 0

        print(f" Detected {len(rpeaks)} R-peaks using BioSPPy")
        print(f" Average HR: {avg_hr:.2f} bpm")

        if plot_peaks:
            plt.figure(figsize=(12, 4))
            plt.plot(times, ecg_data, label='ECG')
            plt.plot(rpeak_times, ecg_data[rpeaks], 'ro', label='R-peaks')
            plt.title("R-Peak Detection (BioSPPy)")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

        # Plot RR intervals
        plt.figure(figsize=(8, 3))
        plt.plot(rr_intervals, marker='o')
        plt.title("RR Intervals Over Time")
        plt.xlabel("Interval Index")
        plt.ylabel("RR Interval (s)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Save RR intervals (optional)
        rr_df = pd.DataFrame({'RR Interval (s)': rr_intervals})
        rr_out_path = os.path.splitext(os.path.basename(self.bdf_path))[0] + "_rr_intervals.csv"
        # rr_df.to_csv(rr_out_path, index=False)
        print(f"📄 RR intervals saved to: {rr_out_path}")

        # Interpolation only if enough peaks to compute HR
        if len(heart_rates) > 0:
            # Heart rate timestamps (midpoints between R-peaks)
            hr_times = (rpeak_times[:-1] + rpeak_times[1:]) / 2

            # Interpolate at 30 Hz using same logic as video frame extraction
            frame_interval = 1.0 / 30.0
            start_sec = int(np.ceil(times[0]))
            end_time = times[-1] - 1e-6  # avoid floating overshoot

            frame_time = start_sec
            frame_times_synced = []
            interpolated_hr_synced = []

            while frame_time <= end_time:
                hr = np.interp(frame_time, hr_times, heart_rates)
                frame_times_synced.append(round(frame_time, 6))  # to match your filename index logic
                interpolated_hr_synced.append(hr)
                frame_time += frame_interval

            df_30hz = pd.DataFrame({
                'Time (s)': frame_times_synced,
                'Heart Rate (bpm)': interpolated_hr_synced
            })
            csv_30hz = os.path.splitext(os.path.basename(self.bdf_path))[0] + "_hr_30hz.csv"
            df_30hz.to_csv(csv_30hz, index=False)
            print(f" Frame-aligned heart rate (30 Hz) saved to: {csv_30hz}")

        return heart_rates, rr_intervals


def robust_r_peak_detection(ecg_data, times):
    fs = int(1 / (times[1] - times[0]))

    # Detect only high-prominence and well-spaced peaks
    threshold = np.percentile(ecg_data, 98)
    peaks, props = find_peaks(ecg_data, distance=fs*0.6, prominence=threshold * 0.3)

    # Calculate heart rate
    rr_intervals = np.diff(times[peaks])
    heart_rates = 60 / rr_intervals if len(rr_intervals) > 0 else []
    avg_hr = np.mean(heart_rates) if heart_rates else 0


    # Plot
    plt.figure(figsize=(12, 4))
    plt.plot(times, ecg_data, label='ECG')
    plt.plot(times[peaks], ecg_data[peaks], 'ro', label='R-peaks')
    plt.title("📈 Robust R-Peak Detection")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f" Cleaned R-peaks: {len(peaks)}")
    print(f" Refined Avg HR: {avg_hr:.2f} bpm")

    return peaks, heart_rates

# === Example Usage ===
if __name__ == "__main__":
    bdf_file = "C:/Users/Hp/Downloads/814/Part_7_S_Trial17_emotion.bdf"
    processor = BDFSyncProcessor(bdf_file)

    processor.print_channels()
    processor.find_trigger_channel()
    processor.detect_first_trigger_time()

    # You can change or add more channels as needed
    channels_to_extract = ['EXG2']
    processor.crop_synced_segment(duration=133.0, channels=channels_to_extract, save_csv=True)
    times, ecg_data = processor.extract_middle_ecg_segment(full_duration_sec=135.0, segment_length_sec=30.0,
                                                           save_csv=False)

    # Preprocess ECG
    filtered_ecg = processor.preprocess_ecg(ecg_data, fs=int(1 / (times[1] - times[0])))
    processor.plot_ecg_signal(times, filtered_ecg, title="Filtered Middle 30s ECG (EXG2)")

    # Heart rate estimation
    processor.estimate_heart_rate(times, filtered_ecg, plot_peaks=True)