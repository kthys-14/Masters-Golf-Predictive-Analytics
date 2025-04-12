import os
import subprocess
import sys

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "imbalanced-learn",
        "streamlit",
        "requests",
        "pillow",
        "pickle-mixin",
        "data-golf"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        for package in missing_packages:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print("All packages installed successfully!")
    else:
        print("All required packages are already installed.")
    
    # Check if streamlit command is available
    try:
        result = subprocess.run(['streamlit', '--version'], capture_output=True, text=True)
        print(f"Streamlit version: {result.stdout.strip()}")
    except FileNotFoundError:
        print("Streamlit command not found in PATH. Fixing...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'streamlit', '--upgrade'])
        print("Please run the application using 'python -m streamlit run golf_predictor.py'")

def create_project_structure():
    """Create the project directory structure"""
    # Create directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("images", exist_ok=True)
    
    print("Project directory structure created!")

def main():
    """Setup the environment"""
    print("Setting up Golf Tournament Winner Predictor...")
    
    check_dependencies()
    create_project_structure()
    
    print("\nSetup complete! To run the application:")
    print("1. Run: streamlit run golf_predictor.py")

if __name__ == "__main__":
    main() 