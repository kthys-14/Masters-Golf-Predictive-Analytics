import os
import subprocess
import sys

def run_app():
    """Run the golf predictor app using Python module approach"""
    print("Starting Golf Predictor Application...")
    
    try:
        # Try running with streamlit module
        subprocess.run([sys.executable, "-m", "streamlit", "run", "golf_predictor.py"], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error running streamlit. Trying to install...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit", "--upgrade"])
        subprocess.run([sys.executable, "-m", "streamlit", "run", "golf_predictor.py"], check=True)

if __name__ == "__main__":
    run_app() 