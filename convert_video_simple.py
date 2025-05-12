import argparse
import os
import subprocess
import shutil
import sys

#python convert_video_simple.py "C:\Users\WuBu\Videos\your_fortnite_video.mp4" "C:\Projects\bytropix\data\demo_video_data_dir\dummy_video.mp4" --width 320 --height 180 --fps 60

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
    gpu_encoder: str = "auto" # "auto", "nvenc", "libx264" (cpu) or other specific encoder
):
    """
    Converts video directly focusing on high quality downscale and target format.
    Output: MP4 container, yuv420p pixel format.
    Uses mpeg4 codec if libx264 is chosen, h264_nvenc if nvenc.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input video file not found: {input_path}")
        return False

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Video filter: scale and set fps.
    # Lanczos is a good quality scaling algorithm.
    vf_options = f"scale={target_width}:{target_height}:flags=lanczos,fps={target_fps}"

    cmd = [
        ffmpeg_exe, "-y",
        "-i", input_path,
        "-vf", vf_options,
        "-an",  # No audio
        "-pix_fmt", "yuv420p", # Crucial for compatibility
    ]

    # Determine encoder and quality settings
    chosen_encoder = ""
    if gpu_encoder == "auto":
        # Basic NVIDIA check
        try:
            encoders_output = subprocess.check_output([ffmpeg_exe, "-encoders"], text=True, stderr=subprocess.DEVNULL)
            if "h264_nvenc" in encoders_output:
                chosen_encoder = "h264_nvenc"
                print("Auto-detected NVIDIA GPU, attempting h264_nvenc.")
            else:
                chosen_encoder = "libx264" # Fallback to CPU mpeg4 via libx264
                print("No NVENC found or auto-detection failed, falling back to libx264 (CPU).")
        except Exception:
            chosen_encoder = "libx264"
            print("Error checking encoders, falling back to libx264 (CPU).")
    elif gpu_encoder == "nvenc":
        chosen_encoder = "h264_nvenc"
    else: # Default or specified CPU
        chosen_encoder = "libx264"


    if chosen_encoder == "h264_nvenc":
        cmd.extend(["-c:v", "h264_nvenc"])
        # For near-lossless with NVENC, use a low QP (Quantization Parameter) value.
        # CRF is not directly available for NVENC in the same way as libx264.
        # -cq or -qp for constant quality. Lower is better. 0 is lossless for some nvenc modes but huge.
        # Let's aim for very high quality, e.g., QP 18-20 (visually near lossless).
        cmd.extend(["-qp", "18"])
        cmd.extend(["-preset", "p7"]) # p7 is slowest/best quality for nvenc, p1 is fastest
        cmd.extend(["-tune", "hq"])   # Tune for high quality
        # cmd.extend(["-rc", "vbr_hq"]) # Variable Bitrate High Quality
        print("Using h264_nvenc with high quality settings (qp 18, preset p7).")
    else: # libx264 (for CPU, can output standard MPEG-4 Part 2 or H.264/MPEG-4 AVC)
        # Your dummy video creation used 'mpeg4' codec directly.
        # If we use libx264, it defaults to H.264 (which is MPEG-4 Part 10).
        # If you strictly need MPEG-4 Part 2 (older, like XVID/DIVX), you'd specify `-c:v mpeg4 -q:v 2`
        # Let's stick to high quality H.264 via libx264 as it's more common for MP4.
        # If your original dummy_video.mp4 was MPEG-4 Part 2, we adjust this.
        # Assuming H.264 is fine for the MP4 container.
        cmd.extend(["-c:v", "libx264"])
        cmd.extend(["-crf", "18"]) # Lower CRF for higher quality (0 is lossless but huge files)
        cmd.extend(["-preset", "slow"]) # Slower preset for better compression/quality
        print("Using libx264 (CPU) with high quality settings (crf 18, preset slow).")


    cmd.append(output_path)

    print(f"\nExecuting FFmpeg command:\n{' '.join(cmd)}\n")

    try:
        # Use DEVNULL for stdout to keep terminal cleaner for a simple conversion script
        process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        _, stderr = process.communicate()
        if process.returncode == 0:
            print(f"Successfully converted video to: {output_path}")
            return True
        else:
            print(f"Error during FFmpeg conversion (return code: {process.returncode}):")
            print("FFmpeg stderr:")
            print(stderr.decode(errors='ignore')) # Show stderr for debugging
            return False
    except FileNotFoundError:
        print(f"Error: ffmpeg command not found at '{ffmpeg_exe}'. Please ensure FFmpeg is installed and in your PATH.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred with FFmpeg: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simple video converter for WuBuNestDiffusion. Downscales and converts to MP4 (H.264/yuv420p) with high quality.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_video", type=str, help="Path to the input WebM or other video file.")
    parser.add_argument("output_video", type=str, help="Path for the output MP4 file (e.g., data/demo_video_data_dir/dummy_video.mp4).")
    parser.add_argument("--width", type=int, default=320, help="Target width.")
    parser.add_argument("--height", type=int, default=180, help="Target height.")
    parser.add_argument("--fps", type=int, default=10, help="Target FPS.")
    parser.add_argument("--ffmpeg_path", type=str, default=None, help="Optional path to ffmpeg executable if not in PATH.")
    parser.add_argument("--encoder", type=str, default="auto", choices=["auto", "nvenc", "cpu"],
                        help="Choose encoder: 'auto' attempts GPU (NVENC), 'nvenc' forces NVIDIA, 'cpu' forces libx264.")

    args = parser.parse_args()

    ffmpeg_executable = args.ffmpeg_path if args.ffmpeg_path else find_ffmpeg()

    if not ffmpeg_executable:
        sys.exit(1)

    gpu_encoder_arg = "auto"
    if args.encoder == "nvenc":
        gpu_encoder_arg = "nvenc"
    elif args.encoder == "cpu":
        gpu_encoder_arg = "libx264" # Explicitly CPU

    success = convert_video_direct(
        args.input_video,
        args.output_video,
        args.width,
        args.height,
        args.fps,
        ffmpeg_exe=ffmpeg_executable,
        gpu_encoder=gpu_encoder_arg
    )

    if success:
        print("Video conversion process finished successfully.")
    else:
        print("Video conversion process encountered errors.")
        sys.exit(1)