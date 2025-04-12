import os
import sys
import subprocess

def install_dependencies():
    """Install required packages for the Golf Predictor app"""
    print("Installing required dependencies...")
    
    # List of required packages
    required_packages = [
        "folium",
        "streamlit-folium"
    ]
    
    # Get the Python executable that's running this script
    python_exe = sys.executable
    
    # Install each package
    for package in required_packages:
        print(f"Installing {package}...")
        try:
            # Try a standard install first
            result = subprocess.run(
                [python_exe, "-m", "pip", "install", package], 
                capture_output=True, 
                text=True
            )
            
            if "already satisfied" in result.stdout:
                print(f"  ✓ {package} is already installed")
            elif result.returncode == 0:
                print(f"  ✓ {package} installed successfully")
            else:
                # If standard install fails, try with --user flag
                print(f"  ! Standard install failed, trying with --user flag")
                result = subprocess.run(
                    [python_exe, "-m", "pip", "install", "--user", package],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    print(f"  ✓ {package} installed successfully with --user flag")
                else:
                    # Last resort: try with break-system-packages
                    print(f"  ! User install failed, trying with --break-system-packages")
                    result = subprocess.run(
                        [python_exe, "-m", "pip", "install", "--user", "--break-system-packages", package],
                        capture_output=True,
                        text=True
                    )
                    
                    if result.returncode == 0:
                        print(f"  ✓ {package} installed with --break-system-packages")
                    else:
                        print(f"  ✗ Failed to install {package}. Error: {result.stderr}")
                        return False
        except Exception as e:
            print(f"  ✗ Error installing {package}: {str(e)}")
            return False
    
    print("All dependencies installed successfully!")
    return True

def run_streamlit_app():
    """Run the Golf Predictor Streamlit app"""
    print("Starting Golf Predictor application...")
    
    # Use the same Python executable to run Streamlit
    python_exe = sys.executable
    
    # Run the Streamlit app
    try:
        # Use the module approach which is more reliable
        subprocess.run(
            [python_exe, "-m", "streamlit", "run", "golf_predictor.py"],
            check=True
        )
    except subprocess.CalledProcessError:
        print("Error running Streamlit app. Please check that streamlit is installed.")
        return False
    except KeyboardInterrupt:
        print("App stopped by user.")
    
    return True

if __name__ == "__main__":
    # Install dependencies first
    if install_dependencies():
        # Then run the app
        run_streamlit_app()
    else:
        print("Failed to install dependencies. Cannot run the app.")
        sys.exit(1) 