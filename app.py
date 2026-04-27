import subprocess
import sys

# main.py ni ishga tushirish
if __name__ == "__main__":
    subprocess.run([
        sys.executable, "-m", "uvicorn", "main:app",
        "--host", "0.0.0.0", "--port", "7860"
    ])