import subprocess
import sys

python_exec = sys.executable  # same interpreter running this script

try:
    print("Running script1...")
    subprocess.run([python_exec, "script1.py"], check=True)

    print("Running script2...")
    subprocess.run([python_exec, "script2.py"], check=True)

    print("Both scripts executed successfully!")

except subprocess.CalledProcessError:
    print("Error: script1 failed, so script2 was NOT executed.")