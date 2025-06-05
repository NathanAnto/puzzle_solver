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
    "sepa_types_final.py",
    "assemblage_json_final_final.py",
    "matching_final_3.py",
    "show_result.py"
]

for script in scripts:
    print(f"Exécution de {script}...")
    subprocess.run(["python", script])
    time.sleep(1)