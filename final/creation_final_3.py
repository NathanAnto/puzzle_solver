import json
import os
from PIL import Image
import numpy as np


def load_puzzle_solution(json_file):
    """Charge la solution du puzzle depuis le fichier JSON."""
    with open(json_file, 'r') as f:
        return json.load(f)


def rotate_image(image, rotation):
    """Applique une rotation à l'image."""
    if rotation == 0:
        return image
    elif rotation == 90:
        return image.rotate(-90, expand=True)
    elif rotation == 180:
        return image.rotate(180, expand=True)
    elif rotation == 270:
        return image.rotate(90, expand=True)
    else:
        return image


def detect_straight_edges(image, threshold=10):
    """
    Détecte les bords droits d'une pièce de puzzle.
    Retourne un dictionnaire indiquant quels côtés sont droits.
    """
    # Convertir en numpy array pour l'analyse
    img_array = np.array(image.convert('RGB'))
    height, width = img_array.shape[:2]

    straight_edges = {'top': False, 'right': False, 'bottom': False, 'left': False}

    # Analyser le bord supérieur
    top_edge = img_array[0, :]
    if is_straight_line(top_edge, threshold):
        straight_edges['top'] = True

    # Analyser le bord droit
    right_edge = img_array[:, -1]
    if is_straight_line(right_edge, threshold):
        straight_edges['right'] = True

    # Analyser le bord inférieur
    bottom_edge = img_array[-1, :]
    if is_straight_line(bottom_edge, threshold):
        straight_edges['bottom'] = True

    # Analyser le bord gauche
    left_edge = img_array[:, 0]
    if is_straight_line(left_edge, threshold):
        straight_edges['left'] = True

    return straight_edges


def is_straight_line(edge_pixels, threshold):
    """
    Vérifie si un bord est une ligne droite en analysant la variance des couleurs.
    Un bord droit aura généralement une couleur uniforme (souvent blanche ou transparente).
    """
    # Calculer la variance des couleurs
    if len(edge_pixels.shape) == 2:  # RGB
        variance = np.var(edge_pixels, axis=0).mean()
    else:  # Grayscale
        variance = np.var(edge_pixels)

    return variance < threshold


def calculate_required_rotation(current_pos, grid_size, straight_edges):
    """
    Calcule la rotation nécessaire pour orienter les bords droits vers l'extérieur.
    """
    row, col = current_pos
    grid_rows, grid_cols = grid_size

    # Déterminer quels côtés doivent être droits selon la position
    required_straight = {
        'top': row == 0,  # Première ligne
        'bottom': row == grid_rows - 1,  # Dernière ligne
        'left': col == 0,  # Première colonne
        'right': col == grid_cols - 1  # Dernière colonne
    }

    # Trouver la rotation qui aligne les bords droits avec l'extérieur
    for rotation in [0, 90, 180, 270]:
        rotated_edges = rotate_edges_mapping(straight_edges, rotation)

        # Vérifier si cette rotation place les bords droits au bon endroit
        match = True
        for direction, should_be_straight in required_straight.items():
            if should_be_straight and not rotated_edges.get(direction, False):
                match = False
                break
            # Optionnel : vérifier qu'aucun bord intérieur n'est droit
            elif not should_be_straight and rotated_edges.get(direction, False):
                match = False
                break

        if match:
            return rotation

    # Si aucune rotation parfaite n'est trouvée, retourner 0
    return 0


def rotate_edges_mapping(edges, rotation):
    """
    Fait tourner le mapping des bords selon la rotation donnée.
    """
    if rotation == 0:
        return edges
    elif rotation == 90:
        return {
            'top': edges['left'],
            'right': edges['top'],
            'bottom': edges['right'],
            'left': edges['bottom']
        }
    elif rotation == 180:
        return {
            'top': edges['bottom'],
            'right': edges['left'],
            'bottom': edges['top'],
            'left': edges['right']
        }
    elif rotation == 270:
        return {
            'top': edges['right'],
            'right': edges['bottom'],
            'bottom': edges['left'],
            'left': edges['top']
        }


def reconstruct_puzzle(solution_file, pieces_folder, output_file, auto_orient_edges=True):
    """
    Reconstruit le puzzle complet à partir de la solution JSON et des pièces.

    Args:
        solution_file: Chemin vers le fichier JSON de solution
        pieces_folder: Dossier contenant les pièces du puzzle
        output_file: Nom du fichier PNG de sortie
        auto_orient_edges: Si True, oriente automatiquement les bords droits vers l'extérieur
    """

    # Charger la solution
    solution = load_puzzle_solution(solution_file)
    grid_rows, grid_cols = solution['grid_size']
    pieces_data = solution['pieces']

    print(f"Reconstruction d'un puzzle {grid_rows}x{grid_cols} avec {len(pieces_data)} pièces...")
    if auto_orient_edges:
        print("Mode auto-orientation des bords activé")

    # Charger la première pièce pour déterminer la taille des pièces
    first_piece_path = os.path.join(pieces_folder, pieces_data[0]['filename'])
    if not os.path.exists(first_piece_path):
        raise FileNotFoundError(f"Pièce non trouvée: {first_piece_path}")

    sample_piece = Image.open(first_piece_path)
    piece_width, piece_height = sample_piece.size
    sample_piece.close()

    print(f"Taille des pièces: {piece_width}x{piece_height}")

    # Créer l'image finale
    final_width = grid_cols * piece_width
    final_height = grid_rows * piece_height
    final_image = Image.new('RGB', (final_width, final_height), (255, 255, 255))

    print(f"Taille de l'image finale: {final_width}x{final_height}")

    # Placer chaque pièce
    pieces_loaded = 0
    rotation_adjustments = 0

    for piece_data in pieces_data:
        filename = piece_data['filename']
        row = piece_data['position']['row']
        col = piece_data['position']['col']
        original_rotation = piece_data['rotation']

        # Charger la pièce
        piece_path = os.path.join(pieces_folder, filename)
        if not os.path.exists(piece_path):
            print(f"Attention: Pièce manquante - {piece_path}")
            continue

        try:
            piece_image = Image.open(piece_path)

            # Déterminer la rotation finale
            final_rotation = original_rotation

            if auto_orient_edges:
                # Détecter les bords droits de la pièce originale
                straight_edges = detect_straight_edges(piece_image)

                # Calculer la rotation nécessaire pour orienter les bords droits vers l'extérieur
                required_rotation = calculate_required_rotation(
                    (row, col), (grid_rows, grid_cols), straight_edges
                )

                # Combiner avec la rotation originale
                final_rotation = (original_rotation + required_rotation) % 360

                if required_rotation != 0:
                    rotation_adjustments += 1
                    print(
                        f"Pièce {filename} à ({row},{col}): rotation ajustée de {required_rotation}° (total: {final_rotation}°)")

            # Appliquer la rotation finale
            if final_rotation != 0:
                piece_image = rotate_image(piece_image, final_rotation)

            # Calculer la position de collage
            x = col * piece_width
            y = row * piece_height

            # Redimensionner la pièce si nécessaire après rotation
            if piece_image.size != (piece_width, piece_height):
                piece_image = piece_image.resize((piece_width, piece_height), Image.LANCZOS)

            # Coller la pièce
            final_image.paste(piece_image, (x, y))
            piece_image.close()

            pieces_loaded += 1
            if pieces_loaded % 5 == 0:
                print(f"Pièces chargées: {pieces_loaded}/{len(pieces_data)}")

        except Exception as e:
            print(f"Erreur lors du chargement de {filename}: {e}")

    # Sauvegarder l'image finale
    final_image.save(output_file, 'PNG', quality=95)
    final_image.close()

    print(f"\nPuzzle reconstruit avec succès!")
    print(f"Image sauvegardée: {output_file}")
    print(f"Pièces utilisées: {pieces_loaded}/{len(pieces_data)}")
    if auto_orient_edges:
        print(f"Rotations automatiques appliquées: {rotation_adjustments}")


def main():
    """Fonction principale."""
    # Configuration des chemins
    solution_file = "puzzle_solution.json"
    pieces_folder = "piece"  # Dossier contenant les pièces
    output_file = "puzzle_complet.png"

    # Vérifier que les fichiers existent
    if not os.path.exists(solution_file):
        print(f"Erreur: Fichier de solution non trouvé - {solution_file}")
        return

    if not os.path.exists(pieces_folder):
        print(f"Erreur: Dossier des pièces non trouvé - {pieces_folder}")
        return

    try:
        # Reconstruction avec auto-orientation des bords
        reconstruct_puzzle(solution_file, pieces_folder, output_file, auto_orient_edges=True)

        # Si vous voulez aussi une version sans auto-orientation pour comparaison :
        # reconstruct_puzzle(solution_file, pieces_folder, "puzzle_original.png", auto_orient_edges=False)

    except Exception as e:
        print(f"Erreur lors de la reconstruction: {e}")


if __name__ == "__main__":
    main()