import argparse
import os
import subprocess
import shutil
import sys
from typing import Optional # Import Optional for type hinting

# Example usage:
# python convert_video_simple.py "input.mp4" "output.mp4" --width 320 --height 180 --fps 10 --duration 15.5
#
# To use NVIDIA GPU encoding (if available):
# python convert_video_simple.py "input.mp4" "output.mp4" --encoder nvenc --duration 10
#
# To force CPU encoding:
# python convert_video_simple.py "input.mp4" "output.mp4" --encoder cpu --duration 10
# python convert_video_simple.py "C:\Users\eman5\Videos\your_fortnite_video.mp4" "C:\Projects\bytropix\data\demo_video_data_dir\dummy_video.mp4" --width 256 --height 256 --fps 60 --encoder nvenc --duration 10

def find_ffmpeg():
    """Tries to find the ffmpeg executable."""
    ffmpeg_exe = shutil.which("ffmpeg")
    if ffmpeg_exe:
        # print(f"Found ffmpeg at: {ffmpeg_exe}")
        return ffmpeg_exe
    else:
        print("ERROR: ffmpeg not found in PATH. Please install ffmpeg and ensure it's in your system PATH.")
        return None

def convert_video_direct(
    input_path: str,
    output_path: str,
    target_width: int,
    target_height: int,
    target_fps: int,
    ffmpeg_exe: str = "ffmpeg",
    gpu_encoder: str = "auto", # "auto", "nvenc", "libx264" (maps to cpu)
    target_duration: Optional[float] = None # Added target_duration
):
    """
    Converts video directly focusing on high quality downscale and target format.
    Output: MP4 container, yuv420p pixel format.
    Uses H.264 codec (via libx264 if CPU, or h264_nvenc if GPU).
    """
    if not os.path.exists(input_path):
        print(f"Error: Input video file not found: {input_path}")
        return False

    output_dir = os.path.dirname(output_path)
    if output_dir: # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

    cmd = [
        ffmpeg_exe, "-y", # -y: overwrite output files without asking
        "-i", input_path,
    ]

    # Add duration if specified. This is an output option, limiting output duration.
    if target_duration is not None and target_duration > 0:
        cmd.extend(["-t", str(target_duration)])
        print(f"Limiting output duration to {target_duration:.2f} seconds.")

    # Video filter: scale and set fps.
    # Lanczos is a good quality scaling algorithm.
    vf_options = f"scale={target_width}:{target_height}:flags=lanczos,fps={target_fps}"

    cmd.extend([
        "-vf", vf_options,
        "-an",  # No audio
        "-pix_fmt", "yuv420p", # Crucial for compatibility
    ])

    # Determine encoder and quality settings
    chosen_encoder = ""
    if gpu_encoder == "auto":
        try:
            # Attempt to check for h264_nvenc availability more directly
            proc_test_nvenc = subprocess.run([ffmpeg_exe, "-hide_banner", "-h", "encoder=h264_nvenc"], capture_output=True, text=True, check=False)
            if proc_test_nvenc.returncode == 0:
                chosen_encoder = "h264_nvenc"
                print("Auto-detected NVIDIA GPU, attempting h264_nvenc.")
            else:
                # Fallback: check full encoders list (slower, more output)
                encoders_output = subprocess.check_output([ffmpeg_exe, "-encoders"], text=True, stderr=subprocess.DEVNULL)
                if "h264_nvenc" in encoders_output:
                    chosen_encoder = "h264_nvenc"
                    print("Auto-detected NVIDIA GPU (fallback check), attempting h264_nvenc.")
                else:
                    chosen_encoder = "libx264"
                    print("No NVENC found or auto-detection failed, falling back to libx264 (CPU).")
        except Exception as e:
            chosen_encoder = "libx264"
            print(f"Error checking encoders ({type(e).__name__}: {e}), falling back to libx264 (CPU).")
    elif gpu_encoder == "nvenc":
        chosen_encoder = "h264_nvenc"
    else: # Default or specified CPU (maps to "libx264")
        chosen_encoder = "libx264"


    if chosen_encoder == "h264_nvenc":
        cmd.extend(["-c:v", "h264_nvenc"])
        # For near-lossless with NVENC, use a low QP (Quantization Parameter) value.
        # QP 18-20 is visually near lossless for many sources.
        cmd.extend(["-qp", "18"])
        # p7 is slowest/best quality for nvenc, p1 is fastest.
        # For newer SDKs, 'slow' (maps to p5-p7 depending on SDK) or 'medium' (p4) are good presets.
        # Using p7 explicitly for max quality if supported. If not, ffmpeg might pick a similar one.
        cmd.extend(["-preset", "p7"]) 
        cmd.extend(["-tune", "hq"])   # Tune for high quality
        print("Using h264_nvenc with high quality settings (qp 18, preset p7, tune hq).")
    else: # libx264 (CPU)
        cmd.extend(["-c:v", "libx264"])
        # Lower CRF for higher quality (0 is lossless but huge files, ~18 is visually lossless)
        cmd.extend(["-crf", "18"]) 
        # Slower preset for better compression/quality.
        cmd.extend(["-preset", "slow"]) 
        print("Using libx264 (CPU) with high quality settings (crf 18, preset slow).")


    cmd.append(output_path)

    print(f"\nExecuting FFmpeg command:\n{' '.join(cmd)}\n")

    try:
        # Use DEVNULL for stdout to keep terminal cleaner for a simple conversion script
        process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        _, stderr_bytes = process.communicate() # stderr is bytes
        stderr_str = stderr_bytes.decode(errors='ignore') # Decode for printing

        if process.returncode == 0:
            print(f"Successfully converted video to: {output_path}")
            # Check if ffmpeg produced any warnings/errors even on success (e.g., "Last message repeated N times")
            if stderr_str.strip():
                 # Show non-critical messages if any, could be warnings or info.
                 # Filter common "frame_drop" messages if too verbose from fps filter
                filtered_stderr = "\n".join(line for line in stderr_str.splitlines() if "Past duration" not in line and "Last message repeated" not in line)
                if filtered_stderr.strip():
                    print("FFmpeg messages (non-critical):")
                    print(filtered_stderr)
            return True
        else:
            print(f"Error during FFmpeg conversion (return code: {process.returncode}):")
            print("FFmpeg stderr:")
            print(stderr_str) 
            return False
    except FileNotFoundError:
        print(f"Error: ffmpeg command not found at '{ffmpeg_exe}'. Please ensure FFmpeg is installed and in your PATH or use --ffmpeg_path.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred with FFmpeg: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simple video converter. Downscales, optionally sets duration, and converts to MP4 (H.264/yuv420p) with high quality.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_video", type=str, help="Path to the input video file.")
    parser.add_argument("output_video", type=str, help="Path for the output MP4 file.")
    parser.add_argument("--width", type=int, default=320, help="Target width.")
    parser.add_argument("--height", type=int, default=180, help="Target height.")
    parser.add_argument("--fps", type=int, default=10, help="Target FPS.")
    parser.add_argument(
        "--duration", 
        type=float, 
        default=None, 
        help="Target duration of the output video in seconds (e.g., 10.5). If not specified or <=0, the whole video is processed."
    )
    parser.add_argument("--ffmpeg_path", type=str, default=None, help="Optional path to ffmpeg executable if not in PATH.")
    parser.add_argument("--encoder", type=str, default="auto", choices=["auto", "nvenc", "cpu"],
                        help="Choose encoder: 'auto' attempts GPU (NVENC if available), 'nvenc' forces NVIDIA GPU, 'cpu' forces libx264 (CPU).")

    args = parser.parse_args()

    ffmpeg_executable = args.ffmpeg_path if args.ffmpeg_path else find_ffmpeg()

    if not ffmpeg_executable:
        sys.exit(1)

    # Map --encoder arg to the internal gpu_encoder parameter convention
    gpu_encoder_param = "auto"
    if args.encoder == "nvenc":
        gpu_encoder_param = "nvenc"
    elif args.encoder == "cpu":
        gpu_encoder_param = "libx264" # Explicitly use libx264 for CPU

    success = convert_video_direct(
        args.input_video,
        args.output_video,
        args.width,
        args.height,
        args.fps,
        ffmpeg_exe=ffmpeg_executable,
        gpu_encoder=gpu_encoder_param,
        target_duration=args.duration 
    )

    if success:
        print("Video conversion process finished successfully.")
    else:
        print("Video conversion process encountered errors.")
        sys.exit(1)