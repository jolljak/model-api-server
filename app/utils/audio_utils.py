import os
import shutil
import subprocess
import tempfile
import soundfile as sf
import math

FFPROBE = "ffprobe"
FFMPEG = "ffmpeg"

def run_cmd(cmd):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = proc.communicate()
    return proc.returncode, out, err

def ensure_tmp_copy(upload_path, file_bytes, suffix):
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    if upload_path:
        shutil.copyfile(upload_path, tmp_path)
    else:
        with open(tmp_path, "wb") as f:
            f.write(file_bytes)
    return tmp_path

def get_media_duration_sec(path):
    code, out, _ = run_cmd([
        FFPROBE, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ])
    if code != 0:
        return None
    try:
        return float(out.strip())
    except:
        return None

def transcode_to_wav_mono16k(src_path):
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    cmd = [
        FFMPEG, "-y",
        "-i", src_path,
        "-ac", "1",
        "-ar", "16000",
        "-f", "wav",
        wav_path,
    ]

    code, out, err = run_cmd(cmd)
    ok = (code == 0 and os.path.exists(wav_path) and os.path.getsize(wav_path) > 0)

    return ok, wav_path, out + err

def is_silent(wav_path, threshold_db=-40.0):
    data, sr = sf.read(wav_path)
    rms = (data**2).mean() ** 0.5
    if rms == 0:
        return True
    db = 20 * math.log10(rms)
    return db < threshold_db
