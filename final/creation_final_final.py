import os
import json
from PIL import Image


def reconstruct_puzzle_from_json(json_path="puzzle_solution_robust.json",
                                 pieces_dir="piece",
                                 output_path="reconstructed_puzzle.png"):
    """
    Reconstruit l'image complète du puzzle à partir d'un fichier JSON de solution.
    - json_path : chemin vers le fichier JSON (ex. "puzzle_solution_robust.json")
    - pieces_dir : dossier contenant les images des pièces (ex. "piece/piece_0.png")
    - output_path: chemin du fichier PNG de sortie

    Le JSON doit contenir une clé "grid", où chaque cellule est soit "VIDE",
    soit "nom_de_fichier:rotation°" (ex. "piece_23.png:90°").
    """

    # 1. Charger le JSON de solution
    with open(json_path, "r") as f:
        data = json.load(f)

    grid = data["grid"]
    grid_height = len(grid)
    grid_width = len(grid[0]) if grid_height > 0 else 0

    if grid_width == 0 or grid_height == 0:
        raise ValueError("Le JSON ne contient pas de grille valide.")

    # 2. Déterminer la taille d'une tuile à partir de la première pièce non-VIDE
    tile_w = tile_h = None
    for row in grid:
        for cell in row:
            if cell != "VIDE":
                filename, rot_str = cell.split(":")
                img_path = os.path.join(pieces_dir, filename)
                if not os.path.isfile(img_path):
                    raise FileNotFoundError(f"Impossible de trouver l'image : {img_path}")
                with Image.open(img_path) as tmp_img:
                    tile_w, tile_h = tmp_img.size
                break
        if tile_w is not None:
            break

    if tile_w is None or tile_h is None:
        raise ValueError("Aucune pièce trouvée dans la grille pour déterminer la taille des tuiles.")

    # 3. Créer une nouvelle image de taille (tile_w * grid_width, tile_h * grid_height)
    full_w = tile_w * grid_width
    full_h = tile_h * grid_height
    # Utilisation d'un fond transparent (RGBA). Si vous préférez un fond blanc, remplacer (255,255,255,0) par (255,255,255).
    full_image = Image.new("RGBA", (full_w, full_h), (255, 255, 255, 0))

    # 4. Parcourir la grille et coller chaque pièce à la bonne position
    for r in range(grid_height):
        for c in range(grid_width):
            cell = grid[r][c]
            if cell == "VIDE":
                # On laisse vide (transparent ou blanc selon le fond choisi)
                continue

            # Ex : "piece_23.png:90°"
            try:
                filename, rot_str = cell.split(":")
            except ValueError:
                raise ValueError(f"Format inattendu dans la grille pour la cellule : '{cell}'")

            angle_deg = int(rot_str.replace("°", "").strip())
            piece_path = os.path.join(pieces_dir, filename)
            if not os.path.isfile(piece_path):
                raise FileNotFoundError(f"Image introuvable pour la pièce : {piece_path}")

            # Charger la pièce
            with Image.open(piece_path).convert("RGBA") as piece_img:
                # Pour une rotation multiple de 90°, on peut utiliser expand=False pour garder la même taille
                if angle_deg != 0:
                    # PIL tourne l'image d'angle degrés COUNTERCLOCKWISE.
                    # Si le JSON indique "90°", on suppose que c'est une rotation dans le sens horaire,
                    # donc on fait rotate(-angle_deg).
                    piece_img = piece_img.rotate(-angle_deg, expand=False)

                # Vérifier que la taille est toujours (tile_w, tile_h)
                if piece_img.size != (tile_w, tile_h):
                    # Si pour une raison quelconque la taille a changé (ex. rectangles),
                    # on redimensionne pour revenir à la taille d'une tuile.
                    piece_img = piece_img.resize((tile_w, tile_h), resample=Image.BICUBIC)

                # Calculer la position de collage
                x = c * tile_w
                y = r * tile_h
                # Coller avec le canal alpha comme masque pour conserver la transparence
                full_image.paste(piece_img, (x, y), piece_img)

    # 5. Sauvegarder l'image finale
    # Si vous voulez un fond blanc plutôt que transparent, vous pouvez convertir en "RGB" sur un blanc :
    # final = Image.new("RGB", (full_w, full_h), (255, 255, 255))
    # final.paste(full_image, mask=full_image.split()[3])  # coller l'alpha
    # final.save(output_path)
    #
    # Ici, on sauve en RGBA :
    full_image.save(output_path)
    print(f"Puzzle reconstruit et sauvegardé sous : {output_path}")


if __name__ == "__main__":
    # Exemple d'utilisation :
    reconstruct_puzzle_from_json(
        json_path="puzzle_solution_robust.json",
        pieces_dir="piece",
        output_path="reconstructed_puzzle.png"
    )
