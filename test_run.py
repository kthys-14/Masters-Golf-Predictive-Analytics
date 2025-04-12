import subprocess
import os
import time
import sys

def run_test():
    """Run a complete test of the golf predictor application"""
    print("=== GOLF PREDICTOR APPLICATION TEST ===")
    
    # Step 1: Check dependencies with test_imports.py
    print("\n1. Testing imports and dependencies...")
    try:
        result = subprocess.run(['python', 'test_imports.py'], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        if "All imports successful" not in result.stdout:
            print("WARNING: Import test may have issues")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Import test failed:\n{e.stderr}")
        return False
    
    # Step 2: Set up environment with setup.py
    print("\n2. Setting up environment...")
    try:
        result = subprocess.run(['python', 'setup.py'], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Setup failed:\n{e.stderr}")
        return False
    
    # Step 3: Launch streamlit in a separate process
    print("\n3. Launching Streamlit application...")
    print("NOTE: Streamlit will run in a separate process. Check your browser.")
    print("      This script will continue to run in the background.")
    
    # Use python -m streamlit instead of streamlit command directly
    streamlit_process = subprocess.Popen(
        [sys.executable, '-m', 'streamlit', 'run', 'golf_predictor.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait a moment for Streamlit to start
    time.sleep(5)
    
    # Check if process is still running
    if streamlit_process.poll() is not None:
        print("ERROR: Streamlit process terminated unexpectedly")
        out, err = streamlit_process.communicate()
        print(f"STDOUT: {out}")
        print(f"STDERR: {err}")
        return False
    
    print("\nApplication is now running. Follow these steps in your browser:")
    print("1. Click 'Fetch Data from DataGolf API'")
    print("2. Click 'Train Prediction Model'")
    print("3. Select a golf course and click 'Predict Tournament Results'")
    print("\nThe application should be available at http://localhost:8501")
    print("\nPress Ctrl+C when you're done testing to terminate this script and the Streamlit process")
    
    try:
        # Keep the script running until user presses Ctrl+C
        while True:
            time.sleep(1)
            # Periodically check streamlit output for errors
            if streamlit_process.poll() is not None:
                print("Streamlit process has terminated")
                break
    except KeyboardInterrupt:
        print("\nTest script terminated by user")
    finally:
        # Clean up the streamlit process
        if streamlit_process.poll() is None:
            print("Terminating Streamlit process...")
            streamlit_process.terminate()
            streamlit_process.wait(timeout=5)
        
        # Collect and display any output from streamlit
        out, err = streamlit_process.communicate()
        if out:
            print("\nApplication output:")
            print(out[:1000])  # Print first 1000 chars to avoid overwhelming the console
            if len(out) > 1000:
                print("... (output truncated)")
        
        if err:
            print("\nApplication errors:")
            print(err)
    
    print("\nTest complete!")
    return True

if __name__ == "__main__":
    run_test() 