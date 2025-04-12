import os
import ast
import sys
import importlib
import inspect

def analyze_code_file(file_path):
    """Analyze a Python file for potential issues"""
    print(f"Analyzing {file_path}...")
    issues = []
    
    # Check if file exists
    if not os.path.exists(file_path):
        issues.append(f"File {file_path} does not exist")
        return issues
    
    try:
        # Read the file
        with open(file_path, 'r') as f:
            code = f.read()
        
        # Parse the code
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            issues.append(f"Syntax error in {file_path}: {str(e)}")
            return issues
        
        # Import the module for inspection
        module_name = os.path.splitext(os.path.basename(file_path))[0]
        try:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Analyze classes and functions
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and obj.__module__ == module_name:
                    # Check class methods
                    for method_name, method in inspect.getmembers(obj, inspect.isfunction):
                        # Check for method signatures and docstrings
                        if method.__doc__ is None:
                            issues.append(f"Missing docstring in {name}.{method_name}")
                        
                        # Check for potential errors in method
                        try:
                            source = inspect.getsource(method)
                            if "except " not in source and ("requests" in source or "client." in source):
                                issues.append(f"API call in {name}.{method_name} may lack error handling")
                        except Exception:
                            pass
            
        except Exception as e:
            issues.append(f"Error importing {module_name}: {str(e)}")
        
        # Look for common issues
        if "st.metric" in code and "st.columns" not in code:
            issues.append("Using st.metric but not in columns which may cause layout issues")
        
        if "model.predict" in code and "try:" not in code:
            issues.append("Model prediction without try/except could crash on invalid input")
            
        # Check for Streamlit session state usage
        if "st.session_state" not in code and "streamlit" in code:
            issues.append("App may have state management issues - st.session_state not used")

        # Check for model persistence
        if "model = None" in code and "pickle.dump" not in code:
            issues.append("Model may not be persisted properly between Streamlit reruns")

        # Check for proper Streamlit widget keys
        if "st.button(" in code and "key=" not in code:
            issues.append("Buttons may need unique keys for proper state management")

        # Check for visualization issues
        if "st.pyplot" in code and "try:" not in code:
            issues.append("Visualization may fail without proper error handling")
        
    except Exception as e:
        issues.append(f"Error analyzing {file_path}: {str(e)}")
    
    return issues

def debug_application():
    """Debug the entire golf predictor application"""
    print("=== GOLF PREDICTOR APPLICATION DEBUG ===")
    
    files_to_check = [
        'golf_predictor.py',
        'test_imports.py',
        'setup.py'
    ]
    
    all_issues = []
    
    for file in files_to_check:
        issues = analyze_code_file(file)
        if issues:
            all_issues.extend([f"[{file}] {issue}" for issue in issues])
            print(f"Found {len(issues)} potential issues in {file}")
        else:
            print(f"No issues found in {file}")
    
    if all_issues:
        print("\nPotential issues found:")
        for i, issue in enumerate(all_issues, 1):
            print(f"{i}. {issue}")
    else:
        print("\nNo potential issues found in the codebase!")
    
    print("\nDebug analysis complete!")

if __name__ == "__main__":
    debug_application() 