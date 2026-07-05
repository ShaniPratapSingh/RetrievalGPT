import os
import sys
import subprocess
import argparse

def check_dependencies():
    """Check if the core dependencies for RetrievalGPT are installed."""
    print("Checking dependencies...")
    missing = []
    
    # Check streamlit
    try:
        import streamlit
        print("✓ streamlit is installed")
    except ImportError:
        missing.append("streamlit")
        print("✗ streamlit is missing")
        
    # Check dotenv
    try:
        import dotenv
        print("✓ python-dotenv is installed")
    except ImportError:
        missing.append("python-dotenv")
        print("✗ python-dotenv is missing")
        
    # Check pypdf
    try:
        import pypdf
        print("✓ pypdf is installed")
    except ImportError:
        missing.append("pypdf")
        print("✗ pypdf is missing")
        
    # Check sentence_transformers
    try:
        import sentence_transformers
        print("✓ sentence-transformers is installed")
    except ImportError:
        missing.append("sentence-transformers")
        print("✗ sentence-transformers is missing (Local embeddings will fall back to hash/mock vectors)")
        


    if missing:
        print("\nSome packages are missing. You can install them by running:")
        print("pip install -r code/requirements.txt")
        print("-" * 50)
        return False
    return True

def setup_env():
    """Copy .env.example to .env if .env doesn't exist."""
    if not os.path.exists(".env"):
        if os.path.exists(".env.example"):
            print("Creating .env from .env.example...")
            try:
                with open(".env.example", "r") as src:
                    content = src.read()
                with open(".env", "w") as dst:
                    dst.write(content)
                print("✓ .env created successfully. Please fill in your API keys.")
            except Exception as e:
                print(f"✗ Failed to copy .env: {e}")
        else:
            print("⚠ .env.example not found. Please create a .env file with your API keys manually.")
    else:
        print("✓ .env file found.")

def main():
    parser = argparse.ArgumentParser(description="Run RetrievalGPT application")
    parser.add_argument("--check-deps", action="store_true", help="Only check dependencies and exit")
    args = parser.parse_args()
    
    # 1. Setup Environment
    setup_env()
    print("-" * 50)
    
    # 2. Check Dependencies
    deps_ok = check_dependencies()
    
    if args.check_deps:
        sys.exit(0 if deps_ok else 1)
        
    # 3. Start App
    app_path = os.path.join("code", "app.py")
    if not os.path.exists(app_path):
        print(f"✗ Error: App file not found at {app_path}")
        sys.exit(1)
        
    print(f"Launching Streamlit application from {app_path}...")
    try:
        # Run streamlit run code/app.py
        subprocess.run(["streamlit", "run", app_path], check=True)
    except FileNotFoundError:
        print("✗ Error: 'streamlit' executable not found. Make sure it's installed and on your PATH.")
        print("Try running: pip install streamlit")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nApplication stopped by user.")
        sys.exit(0)

if __name__ == "__main__":
    main()
