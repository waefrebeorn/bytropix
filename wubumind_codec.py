import librosa
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
from tqdm import tqdm
import soundfile as sf
from matplotlib import cm
from skimage.transform import resize

# ==============================================================================
#   PART 1: THE ZEPHYR-HD ENCODER (1024x1024)
# ==============================================================================
class WubumindCodecZephyrEncoder:
    def __init__(self, sample_rate=48000, duration_secs=4, image_width=1024, image_height=1024, n_fft=2048, debug=True):
        self.sr, self.duration_secs, self.image_width, self.image_height, self.n_fft = sample_rate, duration_secs, image_width, image_height, n_fft
        self.debug = debug
        self.db_range = 140.0
        self.target_len = int(self.sr * self.duration_secs)
        self.side_bar_width = 16
        self.num_bars = 10
        self.bar_height = self.image_height // self.num_bars
        self.feature_width = self.image_width - self.side_bar_width * 2
        
        self.hop_length = self.target_len // self.feature_width
        
        # --- ZEPHYR HARMONIC BANDS ---
        self.band_crossovers_hz = [300, 4000, 10000, 16000] # Bass/Mid, Mid/Presence, Presence/Treble, Treble/Harmonics
        fft_freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        self.band_bins = [0] + [np.searchsorted(fft_freqs, f) for f in self.band_crossovers_hz] + [n_fft // 2 + 1]

    def log(self, message):
        if self.debug:
            print(message)

    def _normalize(self, data, min_v=None, max_v=None):
        min_val = np.min(data) if min_v is None else min_v
        max_val = np.max(data) if max_v is None else max_v
        if max_val - min_val > 1e-6: return (data - min_val) / (max_val - min_val)
        return np.zeros_like(data)

    def generate_from_audio_data(self, y: np.ndarray) -> np.ndarray:
        self.log("\n--- [ZEPHYR-HD DEBUGGER] STARTING ENCODE ---")
        self.log(f"  [INIT] Resolution: {self.image_width}x{self.image_height}")
        canvas = np.zeros((self.image_height, self.image_width, 3), dtype=np.float32)

        original_peak = np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else 1.0
        y = librosa.util.fix_length(y, size=self.target_len)
        
        self.log(f"  [STFT PARAMS] Feature Width: {self.feature_width}, Hop Length: {self.hop_length}")
        
        stft_complex = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
        mag, phase_angle = np.abs(stft_complex), np.angle(stft_complex)
        
        # Ensure exact width by trimming or padding
        if mag.shape[1] > self.feature_width:
            mag = mag[:, :self.feature_width]
            phase_angle = phase_angle[:, :self.feature_width]
        elif mag.shape[1] < self.feature_width:
            pad_width = self.feature_width - mag.shape[1]
            mag = np.pad(mag, ((0, 0), (0, pad_width)), mode='constant')
            phase_angle = np.pad(phase_angle, ((0, 0), (0, pad_width)), mode='constant')

        self.log(f"  [STFT OK] Data shape enforced: {mag.shape}")

        mag_db = librosa.amplitude_to_db(mag, ref=np.max)
        phase_sin, phase_cos = np.sin(phase_angle), np.cos(phase_angle)
        
        columns_slice = slice(self.side_bar_width, -self.side_bar_width)
        
        self.log("  [BANDS] Drawing 5-Band Reversible Spectrum...")
        
        band_definitions = [
            {'name': 'BASS',      'y_start': 0 * self.bar_height, 'y_end': 2 * self.bar_height, 'bin_start': self.band_bins[0], 'bin_end': self.band_bins[1]},
            {'name': 'MIDS',      'y_start': 2 * self.bar_height, 'y_end': 5 * self.bar_height, 'bin_start': self.band_bins[1], 'bin_end': self.band_bins[2]},
            {'name': 'PRESENCE',  'y_start': 5 * self.bar_height, 'y_end': 7 * self.bar_height, 'bin_start': self.band_bins[2], 'bin_end': self.band_bins[3]},
            {'name': 'TREBLE',    'y_start': 7 * self.bar_height, 'y_end': 8 * self.bar_height, 'bin_start': self.band_bins[3], 'bin_end': self.band_bins[4]},
            {'name': 'HARMONICS', 'y_start': 8 * self.bar_height, 'y_end': 9 * self.bar_height, 'bin_start': self.band_bins[4], 'bin_end': self.band_bins[5]}
        ]

        for i, band in enumerate(band_definitions):
            hz_start = self.band_crossovers_hz[i-1] if i > 0 else 0
            hz_end = self.band_crossovers_hz[i] if i < len(self.band_crossovers_hz) else self.sr / 2
            self.log(f"    - Processing {band['name']:<9} ({hz_start:.0f}-{hz_end:.0f} Hz): {band['bin_end'] - band['bin_start']} bins -> {band['y_end'] - band['y_start']} pixels")

            if band['bin_start'] >= band['bin_end']: continue

            target_shape = (band['y_end'] - band['y_start'], self.feature_width)
            
            resized_mag_db = self._normalize(resize(mag_db[band['bin_start']:band['bin_end']], target_shape, anti_aliasing=True), -self.db_range, 0)
            resized_phase_sin = self._normalize(resize(phase_sin[band['bin_start']:band['bin_end']], target_shape, anti_aliasing=True), -1, 1)
            resized_phase_cos = self._normalize(resize(phase_cos[band['bin_start']:band['bin_end']], target_shape, anti_aliasing=True), -1, 1)
            
            canvas[band['y_start']:band['y_end'], columns_slice, 0] = resized_mag_db
            canvas[band['y_start']:band['y_end'], columns_slice, 1] = resized_phase_sin
            canvas[band['y_start']:band['y_end'], columns_slice, 2] = resized_phase_cos
        
        self.log("  [BANDS] Reversible bars complete.")

        self.log("  [METADATA] Drawing final Chroma metadata bar...")
        y_start, y_end = 9 * self.bar_height, self.image_height
        chroma = librosa.feature.chroma_stft(S=mag, sr=self.sr)
        target_shape = (int(y_end - y_start), self.feature_width)
        chroma_resized = resize(self._normalize(chroma), target_shape, anti_aliasing=True)
        canvas[y_start:y_end, columns_slice, :] = cm.hsv(chroma_resized)[:,:,:3]
        self.log("  [METADATA] Metadata complete.")

        avg_profile = self._normalize(np.mean(mag, axis=1))
        profile_bar = resize(avg_profile.reshape(-1, 1), (self.image_height, self.side_bar_width))
        canvas[:, 0:self.side_bar_width, :] = cm.inferno(profile_bar)[:,:,:3]
        canvas[:, -self.side_bar_width:, :] = original_peak
        
        self.log("--- [ZEPHYR-HD DEBUGGER] ENCODE COMPLETE ---")
        return (np.clip(canvas, 0, 1) * 255).astype(np.uint8)

    def generate_from_file(self, audio_path: Path) -> np.ndarray:
        try:
            y, _ = librosa.load(audio_path, sr=self.sr, duration=self.duration_secs, mono=True)
            return self.generate_from_audio_data(y)
        except Exception as e:
            print(f"\nFATAL ERROR processing {audio_path.name}: {type(e).__name__}: {e}\n")
            import traceback
            traceback.print_exc()
            return None

# ==============================================================================
#   PART 2: THE ZEPHYR-HD DECODER (1024x1024)
# ==============================================================================
class WubumindCodecZephyrDecoder:
    def __init__(self, sample_rate=48000, duration_secs=4, image_width=1024, image_height=1024, n_fft=2048, debug=True):
        self.sr, self.duration_secs, self.image_width, self.image_height, self.n_fft = sample_rate, duration_secs, image_width, image_height, n_fft
        self.debug = debug
        self.db_range = 140.0
        self.target_len = int(self.sr * self.duration_secs)
        self.side_bar_width = 16
        self.num_bars = 10
        self.bar_height = self.image_height // self.num_bars
        self.feature_width = self.image_width - self.side_bar_width * 2
        
        self.hop_length = self.target_len // self.feature_width
        
        # --- ZEPHYR HARMONIC BANDS ---
        self.band_crossovers_hz = [300, 4000, 10000, 16000]
        fft_freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        self.band_bins = [0] + [np.searchsorted(fft_freqs, f) for f in self.band_crossovers_hz] + [n_fft // 2 + 1]

    def log(self, message):
        if self.debug:
            print(message)
    
    def _unnormalize(self, data, min_v, max_v):
        return data * (max_v - min_v) + min_v

    def decode_from_image_data(self, image_array_float: np.ndarray) -> np.ndarray:
        self.log("\n--- [ZEPHYR-HD DEBUGGER] STARTING DECODE ---")
        self.log(f"  [INIT] Resolution: {self.image_width}x{self.image_height}")
        
        original_peak = np.mean(image_array_float[0, -self.side_bar_width:, 0])
        self.log(f"  [METADATA] Retrieved original peak amplitude: {original_peak:.4f}")

        num_freq_bins = self.n_fft // 2 + 1
        reconstructed_mag_db = np.zeros((num_freq_bins, self.feature_width))
        reconstructed_phase_sin = np.zeros((num_freq_bins, self.feature_width))
        reconstructed_phase_cos = np.zeros((num_freq_bins, self.feature_width))

        columns_slice = slice(self.side_bar_width, -self.side_bar_width)

        self.log("  [BANDS] Reconstructing 5-Band Spectrum...")
        
        band_definitions = [
            {'name': 'BASS',      'y_start': 0 * self.bar_height, 'y_end': 2 * self.bar_height, 'bin_start': self.band_bins[0], 'bin_end': self.band_bins[1]},
            {'name': 'MIDS',      'y_start': 2 * self.bar_height, 'y_end': 5 * self.bar_height, 'bin_start': self.band_bins[1], 'bin_end': self.band_bins[2]},
            {'name': 'PRESENCE',  'y_start': 5 * self.bar_height, 'y_end': 7 * self.bar_height, 'bin_start': self.band_bins[2], 'bin_end': self.band_bins[3]},
            {'name': 'TREBLE',    'y_start': 7 * self.bar_height, 'y_end': 8 * self.bar_height, 'bin_start': self.band_bins[3], 'bin_end': self.band_bins[4]},
            {'name': 'HARMONICS', 'y_start': 8 * self.bar_height, 'y_end': 9 * self.bar_height, 'bin_start': self.band_bins[4], 'bin_end': self.band_bins[5]}
        ]

        for band in band_definitions:
            self.log(f"    - Reconstructing {band['name']:<9}: {band['y_end'] - band['y_start']} pixels -> {band['bin_end'] - band['bin_start']} bins")
            if band['bin_start'] >= band['bin_end']: continue
            
            bar_mag_db_norm = image_array_float[band['y_start']:band['y_end'], columns_slice, 0]
            bar_phase_sin_norm = image_array_float[band['y_start']:band['y_end'], columns_slice, 1]
            bar_phase_cos_norm = image_array_float[band['y_start']:band['y_end'], columns_slice, 2]

            bar_mag_db = self._unnormalize(bar_mag_db_norm, -self.db_range, 0)
            bar_phase_sin = self._unnormalize(bar_phase_sin_norm, -1, 1)
            bar_phase_cos = self._unnormalize(bar_phase_cos_norm, -1, 1)

            target_shape = (band['bin_end'] - band['bin_start'], self.feature_width)
            resized_mag_db = resize(bar_mag_db, target_shape, anti_aliasing=True)
            resized_phase_sin = resize(bar_phase_sin, target_shape, anti_aliasing=True)
            resized_phase_cos = resize(bar_phase_cos, target_shape, anti_aliasing=True)
            
            reconstructed_mag_db[band['bin_start']:band['bin_end'], :] = resized_mag_db
            reconstructed_phase_sin[band['bin_start']:band['bin_end'], :] = resized_phase_sin
            reconstructed_phase_cos[band['bin_start']:band['bin_end'], :] = resized_phase_cos
        
        self.log("  [BANDS] STFT band reconstruction complete.")
        
        reconstructed_mag = librosa.db_to_amplitude(reconstructed_mag_db)
        reconstructed_phase_angle = np.arctan2(reconstructed_phase_sin, reconstructed_phase_cos)
        stft_complex = reconstructed_mag * np.exp(1j * reconstructed_phase_angle)
        
        y = librosa.istft(stft_complex, n_fft=self.n_fft, hop_length=self.hop_length, length=self.target_len)
        self.log("  [ISTFT] Inverse transform complete.")
        
        current_peak = np.max(np.abs(y))
        if current_peak > 1e-6:
            y = y * (original_peak / current_peak)
        
        self.log("--- [ZEPHYR-HD DEBUGGER] DECODE COMPLETE ---")
        return y

    def decode_from_file(self, image_path: Path) -> np.ndarray:
        try:
            with Image.open(image_path) as img:
                img_array = np.array(img.convert("RGB"))
                img_array_float = img_array.astype(np.float32) / 255.0
                return self.decode_from_image_data(img_array_float)
        except Exception as e:
            print(f"\nFATAL ERROR processing {image_path.name}: {type(e).__name__}: {e}\n")
            import traceback
            traceback.print_exc()
            return None

# ==============================================================================
#   PART 3: COMMAND-LINE INTERFACE
# ==============================================================================
def main_cli():
    parser = argparse.ArgumentParser(description="Wubumind Visual Audio Codec v10 (ZEPHYR-HD)")
    # --- NEW: Configurable Resolution ---
    parser.add_argument('--res', type=int, default=1024, help="Set the image resolution (e.g., 512, 1024).")
    parser.add_argument('--no-debug', action='store_true', help="Disable verbose debug logging.")
    subparsers = parser.add_subparsers(dest='command', required=True)
    parser_encode = subparsers.add_parser('encode')
    parser_encode.add_argument("input_path", type=str)
    parser_encode.add_argument("output_dir", type=str)
    parser_decode = subparsers.add_parser('decode')
    parser_decode.add_argument("input_path", type=str)
    parser_decode.add_argument("output_dir", type=str)
    args = parser.parse_args()

    codec_params = {
        'image_width': args.res,
        'image_height': args.res,
        'debug': not args.no_debug
    }

    if args.command == 'encode':
        input_path, output_dir = Path(args.input_path), Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(list(input_path.glob('*.mp3')) + list(input_path.glob('*.wav')) + list(input_path.glob('*.flac'))) if input_path.is_dir() else [input_path]
        encoder = WubumindCodecZephyrEncoder(**codec_params)
        for file in tqdm(files, desc="Encoding Audio (ZEPHYR-HD)"):
            img_array = encoder.generate_from_file(file)
            if img_array is not None:
                Image.fromarray(img_array).save(output_dir / f"{file.stem}_zephyr_hd_{args.res}.png")
        print(f"\n✅ Encoding complete! Images saved to {output_dir}")

    elif args.command == 'decode':
        input_path, output_dir = Path(args.input_path), Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(list(input_path.glob('*.png'))) if input_path.is_dir() else [input_path]
        decoder = WubumindCodecZephyrDecoder(**codec_params)
        for file in tqdm(files, desc="Decoding Images (ZEPHYR-HD)"):
            audio_data = decoder.decode_from_file(file)
            if audio_data is not None:
                sf.write(output_dir / f"{file.stem}_decoded.wav", audio_data, decoder.sr)
        print(f"\n✅ Decoding complete! Audio saved to {output_dir}")

if __name__ == "__main__":
    main_cli()