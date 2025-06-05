import os
import json
from PIL import Image


def assemble_puzzle(solution_json_path: str,
                    pieces_folder: str,
                    output_path: str = "puzzle_final.png"):
    """
    Lit le JSON de solution, charge chaque image de pièce depuis `pieces_folder`,
    la tourne selon l'angle indiqué, puis la colle à la bonne position dans un grand canevas.

    Les chemins `solution_json_path` et `pieces_folder` sont relatifs au dossier
    où se situe ce script Python (grâce à __file__).
    """
    # On récupère le dossier où se trouve ce script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Chemin absolu vers le JSON
    json_abspath = os.path.join(script_dir, solution_json_path)
    if not os.path.isfile(json_abspath):
        raise FileNotFoundError(f"Le fichier JSON n'a pas été trouvé : {json_abspath}")

    # Charger le JSON de solution
    with open(json_abspath, "r") as f:
        solution = json.load(f)

    # On s'attend à avoir :
    #   solution["grid_size"] = [rows, cols]
    #   solution["pieces"] = liste d'objets { "filename": ..., "position": {"row":r,"col":c}, "rotation": deg }
    rows, cols = solution["grid_size"]
    placements = solution["pieces"]

    # 1) Déterminer la taille d'une pièce : on en prend une au hasard
    if len(placements) == 0:
        raise ValueError("Aucune pièce à placer dans le JSON.")

    sample_filename = placements[0]["filename"]
    sample_path = os.path.join(script_dir, pieces_folder, sample_filename)
    if not os.path.isfile(sample_path):
        raise FileNotFoundError(f"Impossible de trouver la pièce de référence : {sample_path}")

    sample_img = Image.open(sample_path).convert("RGBA")
    piece_w, piece_h = sample_img.size

    # 2) Créer un canevas vide (transparent) de taille (cols * piece_w) x (rows * piece_h)
    canvas_w = cols * piece_w
    canvas_h = rows * piece_h
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))

    # 3) Pour chaque placement, charger, tourner et coller au bon endroit
    for entry in placements:
        fname = entry["filename"]
        r = entry["position"]["row"]
        c = entry["position"]["col"]
        rot = entry["rotation"]  # en degrés dans le sens horaire

        piece_path = os.path.join(script_dir, pieces_folder, fname)
        if not os.path.isfile(piece_path):
            print(f"⚠️  Pièce non trouvée, on passe : {piece_path}")
            continue

        img = Image.open(piece_path).convert("RGBA")
        # PIL tourne anti-horaire par défaut, donc on fait -rot pour un rot horaire.
        img_rotated = img.rotate(-rot, expand=False)

        # Calculer la position (x0, y0) dans le canevas
        x0 = c * piece_w
        y0 = r * piece_h

        # Coller en conservant l'alpha
        canvas.paste(img_rotated, (x0, y0), img_rotated)

    # 4) Sauvegarder l'image finale
    output_abspath = os.path.join(script_dir, output_path)
    canvas.save(output_abspath)
    print(f"✅ Puzzle assemblé et sauvegardé dans « {output_abspath} ».")


if __name__ == "__main__":
    # Exemple d'utilisation :
    # - puzzle_solution.json et dossier "piece/" doivent être dans le même dossier que ce script.
    assemble_puzzle(
        solution_json_path="puzzle_solution.json",
        pieces_folder="piece",
        output_path="puzzle_final.png"
    )
