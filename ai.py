import subprocess
import sys
import platform

def install_packages():
    """Install all required packages for the job portal application"""
    required_packages = [
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "tk"  # Tkinter (usually comes with Python)
    ]

    print(f"Python {sys.version.split()[0]} on {platform.system()}")
    print("Installing required packages...\n")

    for package in required_packages:
        try:
            # Special handling for Tkinter
            if package == "tk":
                if platform.system() == "Linux":
                    print("Installing python3-tk for Linux...")
                    subprocess.run(["sudo", "apt-get", "install", "python3-tk"], check=True)
                continue
            
            print(f"Installing {package}...", end=" ", flush=True)
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print("Done")
        except subprocess.CalledProcessError as e:
            print(f"\nFailed to install {package}: {e}")
        except Exception as e:
            print(f"\nError installing {package}: {e}")

    print("\nAll packages installed successfully!")
    print("Now you can run the main application (job_portal.py)")

if __name__ == "__main__":
    install_packages()