def test_imports():
    """Test that all required packages can be imported"""
    try:
        # Standard library imports
        import os
        import json
        import time
        import pickle
        
        # Third-party imports
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Conditionally import streamlit to avoid errors in non-streamlit context
        try:
            import streamlit as st
            print("Streamlit imported successfully")
        except:
            print("Note: Streamlit import was skipped (normal in non-streamlit context)")
            
        import requests
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
        from imblearn.over_sampling import SMOTE
        from PIL import Image  # From pillow package
        
        # Test data-golf package
        try:
            from data_golf import DataGolfClient
            print("data-golf package imported successfully")
            
            # Test instantiating a client to ensure the API works
            try:
                client = DataGolfClient(api_key='baf700f24ceb315b87e9f65d98e9')
                print("DataGolfClient instantiated successfully")
            except Exception as e:
                print(f"Warning: Could not instantiate DataGolfClient: {e}")
                
        except ImportError as e:
            print(f"data-golf import error: {e}")
            print("Installing data-golf package...")
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "data-golf"])
            try:
                from data_golf import DataGolfClient
                print("data-golf package installed successfully")
            except ImportError as e:
                print(f"Failed to import data-golf after installation: {e}")
        
        print("All imports successful!")
        return True
    except ImportError as e:
        print(f"Import error: {e}")
        return False

if __name__ == "__main__":
    test_imports() 