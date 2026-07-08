import subprocess
import sys

if __name__ == "__main__":
    print("Running Classification.py ...")
    subprocess.run([sys.executable, "Classification.py"], check=True)

    print("\nClassification done.\n")
    print("Running Staining.py ...")
    subprocess.run([sys.executable, "Staining.py"], check=True)

    print("\nPipeline complete: Classification + Staining finished successfully.")
