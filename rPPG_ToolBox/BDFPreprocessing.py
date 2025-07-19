import chardet
import mne
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.signal import find_peaks
from biosppy.signals import ecg as biosppy_ecg
import xml.etree.ElementTree as ET


class BDFSyncProcessor:
    def __init__(self, bdf_path,xml_path):
        self.bdf_path = bdf_path
        self.xml_path = xml_path
        self.raw = mne.io.read_raw_bdf(bdf_path, preload=True)

        self.trigger_channel = None
        self.trigger_time = None
        self.trigger_value = None

    import xml.etree.ElementTree as ET

    def extract_xml_sync_params(self):
        tree = ET.parse(self.xml_path)
        root = tree.getroot()

        vid_begin = float(root.attrib['vidBeginSmp'])
        vid_end = float(root.attrib['vidEndSmp'])
        vid_rate = float(root.attrib['vidRate'])

        return vid_begin, vid_end, vid_rate

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

    def extract_trigger_signal(self, save_csv=True):
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

    def extract_middle_ecg_segment(self, csv_path=None, segment_length_sec=30.0, save_csv=True):
        """
        Reads the previously saved synced ECG CSV file, finds the middle 30s segment,
        and returns time and ECG data for heart rate analysis.
        """
        if csv_path is None:
            # Default CSV name based on BDF filename
            csv_path = os.path.splitext(os.path.basename(self.bdf_path))[0] + "_synced_data.csv"

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Synced CSV file not found: {csv_path}")

        # Read the full ECG data
        df = pd.read_csv(csv_path)

        # Validate structure
        if 'Time (s)' not in df.columns or 'EXG2' not in df.columns:
            raise ValueError("CSV must contain 'Time (s)' and 'EXG2' columns")

        times = df['Time (s)'].values
        ecg_data = df['EXG2'].values

        full_duration = times[-1] - times[0]
        print(f" Full Synced ECG Duration: {full_duration:.2f} seconds")

        # Determine start and end of middle segment
        mid_point = times[0] + full_duration / 2
        t_start = mid_point - (segment_length_sec / 2)
        t_end = mid_point + (segment_length_sec / 2)

        # Mask the segment
        mask = (times >= t_start) & (times <= t_end)
        segment_times = times[mask]
        segment_ecg = ecg_data[mask]

        print(f" Extracted Middle {segment_length_sec}s ECG Segment: Start={t_start:.2f}s, End={t_end:.2f}s")

        # Save segment if required
        if save_csv:
            df_segment = pd.DataFrame({'Time (s)': segment_times, 'EXG2 (µV)': segment_ecg})
            out_name = os.path.splitext(csv_path)[0] + f"_middle_{int(segment_length_sec)}s.csv"
            df_segment.to_csv(out_name, index=False)
            print(f" Saved to: {out_name}")

        return segment_times, segment_ecg

    def crop_synced_segment(self, duration, channels=None, save_csv=False):
        if self.trigger_time is None:
            raise RuntimeError("Trigger time not set. Run detect_first_trigger_time() first.")


        end_time = self.trigger_time + duration
        cropped_raw = self.raw.copy()

        if channels:
            cropped_raw.pick(channels)

        cropped_raw.crop(tmin=self.trigger_time, tmax=end_time)
        data, times = cropped_raw.get_data(return_times=True)

        # Save CSV if needed
        if save_csv:
            df = pd.DataFrame({'Time (s)': times})
            for i, ch in enumerate(cropped_raw.ch_names):
                df[ch] = data[i]
            out_name = os.path.splitext(os.path.basename(self.bdf_path))[0] + "_synced_data.csv"
            df.to_csv(out_name, index=False)
            print(f" Synced data saved to {out_name}")

        return cropped_raw

    def plot_ecg_signal(self, times, ecg_data, title="Extracted ECG (EXG2)"):
        """
        Plots the ECG signal (EXG2) over time and prints time range and duration.
        """
        start_time = times[0]
        end_time = times[-1]
        duration = end_time - start_time

        print(f" Segment Start Time: {start_time:.3f} s")
        print(f" Segment End Time:   {end_time:.3f} s")
        print(f"️ Segment Duration:   {duration:.3f} s")

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



    def calculate_video_duration(self):
        begin_sample, end_sample, vid_rate = self.extract_xml_sync_params()
        return (end_sample - begin_sample) / vid_rate



# === Example Usage ===
if __name__ == "__main__":
    folderPath = "C:/Users/Hp/Downloads/2618/"
    bdf_file = os.path.join(folderPath, "Part_21_S_Trial9_emotion.bdf")
    xml_file = os.path.join(folderPath, "session.xml")
    processor = BDFSyncProcessor(bdf_file,xml_file)

    processor.print_channels()
    processor.find_trigger_channel()
    processor.detect_first_trigger_time()
    video_duration = processor.calculate_video_duration()

    # You can change or add more channels as needed
    channels_to_extract = ['EXG2']
    processor.crop_synced_segment(video_duration, channels=channels_to_extract, save_csv=True)
    times, ecg_data = processor.extract_middle_ecg_segment( segment_length_sec=30.0,
                                                           save_csv=True)

    # Preprocess ECG
    filtered_ecg = processor.preprocess_ecg(ecg_data, fs=int(1 / (times[1] - times[0])))
    processor.plot_ecg_signal(times, filtered_ecg, title="Filtered Middle 30s ECG (EXG2)")

    # Heart rate estimation
    processor.estimate_heart_rate(times, filtered_ecg, plot_peaks=True)