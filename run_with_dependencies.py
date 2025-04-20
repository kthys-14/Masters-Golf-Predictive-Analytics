import os
import sys
import subprocess

def install_dependencies():
    """Install required packages for the Golf Predictor app"""
    print("Installing required dependencies...")
    
    # List of required packages
    required_packages = [
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0",
        "imbalanced-learn>=0.8.0",
        "streamlit>=1.13.0",
        "requests>=2.26.0",
        "pillow>=9.0.0",
        "pickle-mixin>=1.0.2",
        "data-golf>=0.5.1",
        "folium>=0.12.1",
        "streamlit-folium>=0.7.0"
    ]
    
    # Get the Python executable that's running this script
    python_exe = sys.executable
    
    # Install each package
    for package_spec in required_packages:
        # Extract the package name without version
        package = package_spec.split('>=')[0].split('==')[0].strip()
        
        print(f"Installing {package_spec}...")
        try:
            # Try a standard install first
            result = subprocess.run(
                [python_exe, "-m", "pip", "install", package_spec], 
                capture_output=True, 
                text=True
            )
            
            if "already satisfied" in result.stdout:
                print(f"  ✓ {package_spec} is already installed")
            elif result.returncode == 0:
                print(f"  ✓ {package_spec} installed successfully")
            else:
                # If standard install fails, try with --user flag
                print(f"  ! Standard install failed, trying with --user flag")
                result = subprocess.run(
                    [python_exe, "-m", "pip", "install", "--user", package_spec],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    print(f"  ✓ {package_spec} installed successfully with --user flag")
                else:
                    # Last resort: try with break-system-packages
                    print(f"  ! User install failed, trying with --break-system-packages")
                    result = subprocess.run(
                        [python_exe, "-m", "pip", "install", "--user", "--break-system-packages", package_spec],
                        capture_output=True,
                        text=True
                    )
                    
                    if result.returncode == 0:
                        print(f"  ✓ {package_spec} installed with --break-system-packages")
                    else:
                        print(f"  ✗ Failed to install {package_spec}. Error: {result.stderr}")
                        return False
        except Exception as e:
            print(f"  ✗ Error installing {package_spec}: {str(e)}")
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