import subprocess
import time

scripts = [
    "detouring.py",
    "dectection_cote.py",
    "detect_flood_filling.py",
    "corner_detect.py",
    "supp_color_font.py"
]

for script in scripts:
    print(f"Exécution de {script}...")
    subprocess.run(["python", script])
    time.sleep(1)  # Pause de 1 seconde
