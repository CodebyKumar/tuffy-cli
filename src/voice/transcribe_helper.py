import sys
import os

def suppress_stdout_stderr():
    """Suppresses all console outputs, including low-level C library writes."""
    try:
        null_fd = os.open(os.devnull, os.O_RDWR)
        save_stdout = os.dup(1)
        save_stderr = os.dup(2)
        os.dup2(null_fd, 1)
        os.dup2(null_fd, 2)
        return save_stdout, save_stderr, null_fd
    except Exception:
        return None

def restore_stdout_stderr(saved):
    if saved:
        save_stdout, save_stderr, null_fd = saved
        os.dup2(save_stdout, 1)
        os.dup2(save_stderr, 2)
        os.close(save_stdout)
        os.close(save_stderr)
        os.close(null_fd)

def main():
    if len(sys.argv) < 4:
        print("Usage: transcribe_helper.py <wav_path> <model_name> <models_dir>")
        sys.exit(1)
        
    wav_path = sys.argv[1]
    model_name = sys.argv[2]
    models_dir = sys.argv[3]
    
    # Check if running on Jetson Orin to use GPU acceleration
    is_jetson = os.path.exists("/etc/nv_tegra_release")
    
    # Suppress output while importing and initializing
    saved = suppress_stdout_stderr()
    try:
        from pywhispercpp.model import Model
        
        params = {}
        if is_jetson:
            params["context_params"] = {"use_gpu": True}
            
        model = Model(model=model_name, models_dir=models_dir, **params)
        segments = model.transcribe(wav_path, print_progress=False)
        transcription = " ".join(segment.text.strip() for segment in segments).strip()
    except Exception as e:
        restore_stdout_stderr(saved)
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
        
    restore_stdout_stderr(saved)
    print(transcription)

if __name__ == "__main__":
    main()
