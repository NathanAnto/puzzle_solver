import subprocess
import time

scripts = [
    "enleve_fond_final.py",
    "noir_blanc_final.py",
    "separation_final.py",
    "extra_coins_final.py",
    "sepa_normalise_finale.py",
    "sepa_couleur_final.py",
    "extra_couleur_final.py",
    "sepa_types.py.py",
    "assemblage_json_final_final.py",
    "matching_final_3.py",
    "creation_final_3.py"
]

for script in scripts:
    print(f"Exécution de {script}...")
    subprocess.run(["python", script])
    time.sleep(1)