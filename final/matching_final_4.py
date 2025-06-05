import numpy as np
import json
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.spatial.distance import euclidean
import cv2


@dataclass
class PieceInfo:
    """Information sur une pièce de puzzle"""
    filename: str
    rotation: int  # 0, 90, 180, 270 degrés
    position: Tuple[int, int]  # (row, col) dans la grille


@dataclass
class MatchScore:
    """Score de matching entre deux côtés"""
    piece1: str
    side1: str
    piece2: str
    side2: str
    color_score: float
    shape_score: float
    total_score: float
    is_valid: bool


class PuzzleMatcher:
    def __init__(self, puzzle_data_path: str, grid_size: Tuple[int, int] = (4, 6)):
        """
        Initialise le matcher de puzzle

        Args:
            puzzle_data_path: Chemin vers le fichier JSON contenant toutes les données
            grid_size: Taille de la grille (rows, cols) - default (4, 6) pour 24 pièces
        """
        self.grid_rows, self.grid_cols = grid_size
        self.total_pieces = self.grid_rows * self.grid_cols

        # Charger les données
        with open(puzzle_data_path, 'r') as f:
            self.puzzle_data = json.load(f)

        print(f"Données chargées pour {len(self.puzzle_data)} pièces")

        # Poids pour le scoring
        self.COLOR_WEIGHT = 0.9
        self.SHAPE_WEIGHT = 0.1

        # Mappings des côtés selon la rotation
        self.side_rotation_map = {
            0: {'haut': 'haut', 'droite': 'droite', 'bas': 'bas', 'gauche': 'gauche'},
            90: {'haut': 'gauche', 'droite': 'haut', 'bas': 'droite', 'gauche': 'bas'},
            180: {'haut': 'bas', 'droite': 'gauche', 'bas': 'haut', 'gauche': 'droite'},
            270: {'haut': 'droite', 'droite': 'bas', 'bas': 'gauche', 'gauche': 'haut'}
        }

        # Grille de solution
        self.grid = [[None for _ in range(self.grid_cols)] for _ in range(self.grid_rows)]
        self.placed_pieces = set()

        # Cache des scores pour optimisation
        self.score_cache = {}

        # Classifier les pièces par type
        self.classify_pieces()

    def classify_pieces(self):
        """Classifie les pièces en coins, bordures et intérieures"""
        self.corner_pieces = []
        self.border_pieces = []
        self.interior_pieces = []

        for piece in self.puzzle_data:
            neutral_count = sum(1 for side in ['haut', 'droite', 'bas', 'gauche']
                                if piece['sides'][side]['gender'] == 'neutre')

            if neutral_count == 2:
                self.corner_pieces.append(piece['filename'])
            elif neutral_count == 1:
                self.border_pieces.append(piece['filename'])
            elif neutral_count == 0:
                self.interior_pieces.append(piece['filename'])
            else:
                print(f"Attention: pièce {piece['filename']} avec {neutral_count} côtés neutres")

        print(f"Classification: {len(self.corner_pieces)} coins, "
              f"{len(self.border_pieces)} bordures, {len(self.interior_pieces)} intérieures")

    def get_piece_data(self, piece_name: str) -> Dict:
        """Récupère les données d'une pièce"""
        for piece in self.puzzle_data:
            if piece['filename'] == piece_name:
                return piece
        return None

    def get_rotated_side(self, side: str, rotation: int) -> str:
        """Retourne le nom du côté après rotation"""
        return self.side_rotation_map[rotation][side]

    def get_neutral_sides(self, piece_data: Dict) -> List[str]:
        """Retourne les côtés neutres d'une pièce"""
        return [side for side in ['haut', 'droite', 'bas', 'gauche']
                if piece_data['sides'][side]['gender'] == 'neutre']

    def get_required_neutral_sides(self, row: int, col: int) -> List[str]:
        """Retourne les côtés qui doivent être neutres pour une position donnée"""
        required_neutrals = []

        if row == 0:
            required_neutrals.append('haut')
        if row == self.grid_rows - 1:
            required_neutrals.append('bas')
        if col == 0:
            required_neutrals.append('gauche')
        if col == self.grid_cols - 1:
            required_neutrals.append('droite')

        return required_neutrals

    def find_valid_rotation(self, piece_name: str, row: int, col: int) -> Optional[int]:
        """
        Trouve une rotation valide pour placer une pièce à une position donnée
        en s'assurant que les côtés neutres pointent vers l'extérieur
        """
        piece_data = self.get_piece_data(piece_name)
        if not piece_data:
            return None

        neutral_sides = self.get_neutral_sides(piece_data)
        required_neutrals = self.get_required_neutral_sides(row, col)

        # Vérifier si la pièce a le bon nombre de côtés neutres
        if len(neutral_sides) != len(required_neutrals):
            return None

        # Essayer toutes les rotations
        for rotation in [0, 90, 180, 270]:
            rotated_neutrals = [self.get_rotated_side(side, rotation) for side in neutral_sides]

            # Vérifier si tous les côtés neutres requis sont bien neutres après rotation
            if set(rotated_neutrals) == set(required_neutrals):
                # Vérifier aussi que les côtés non-neutres ne pointent pas vers l'extérieur
                all_valid = True

                # Vérifier que les côtés intérieurs ne sont pas neutres
                if row > 0 and self.get_rotated_side('haut', rotation) in neutral_sides:
                    if 'haut' not in required_neutrals:
                        all_valid = False
                if row < self.grid_rows - 1 and self.get_rotated_side('bas', rotation) in neutral_sides:
                    if 'bas' not in required_neutrals:
                        all_valid = False
                if col > 0 and self.get_rotated_side('gauche', rotation) in neutral_sides:
                    if 'gauche' not in required_neutrals:
                        all_valid = False
                if col < self.grid_cols - 1 and self.get_rotated_side('droite', rotation) in neutral_sides:
                    if 'droite' not in required_neutrals:
                        all_valid = False

                if all_valid:
                    return rotation

        return None

    def calculate_color_similarity(self, colors1: List[Dict], colors2: List[Dict]) -> float:
        """
        Calcule la similarité entre deux séries de couleurs
        Les couleurs doivent être inversées pour le matching
        """
        if not colors1 or not colors2:
            return 0.0

        total_distance = 0
        count = 0

        # Inverser colors2 pour le matching
        colors2_reversed = list(reversed(colors2))

        # Comparer chaque segment
        min_segments = min(len(colors1), len(colors2_reversed))

        for i in range(min_segments):
            color1 = np.array(colors1[i]['color_rgb'])
            color2 = np.array(colors2_reversed[i]['color_rgb'])

            # Distance euclidienne dans l'espace RGB
            distance = euclidean(color1, color2)
            total_distance += distance
            count += 1

        if count == 0:
            return 0.0

        # Normaliser la distance (0-255*sqrt(3) -> 0-1)
        avg_distance = total_distance / count
        max_distance = 255 * np.sqrt(3)

        # Convertir en similarité (1 = parfait, 0 = très différent)
        similarity = 1.0 - (avg_distance / max_distance)

        return similarity

    def calculate_shape_similarity(self, points1: List, points2: List) -> float:
        """
        Calcule la similarité entre deux formes normalisées
        Version simplifiée pour éviter les erreurs
        """
        if not points1 or not points2 or len(points1) != 17 or len(points2) != 17:
            return 0.5  # Score neutre par défaut

        try:
            # Convertir en numpy arrays
            pts1 = np.array(points1)
            pts2 = np.array(points2)

            # Inverser pts2 pour le matching
            pts2_reversed = pts2[::-1]

            # Calculer la différence des aires sous les courbes
            area1 = float(pts1[-1][1])  # Utiliser l'aire stockée si disponible
            area2 = float(pts2_reversed[-1][1])

            # Si les aires ont des signes opposés, c'est bon pour le matching
            if (area1 > 0 and area2 < 0) or (area1 < 0 and area2 > 0):
                # Plus les valeurs absolues sont proches, meilleur est le score
                diff = abs(abs(area1) - abs(area2))
                max_area = max(abs(area1), abs(area2))
                if max_area > 0:
                    similarity = 1.0 - (diff / max_area)
                else:
                    similarity = 0.5
            else:
                # Mauvais matching si les signes sont identiques
                similarity = 0.1

            return max(0.0, min(1.0, similarity))

        except Exception as e:
            # En cas d'erreur, retourner un score neutre
            return 0.5

    def check_gender_compatibility(self, gender1: str, gender2: str) -> bool:
        """Vérifie la compatibilité des genres"""
        if gender1 == 'male' and gender2 == 'femelle':
            return True
        if gender1 == 'femelle' and gender2 == 'male':
            return True
        if gender1 == 'neutre' or gender2 == 'neutre':
            return False
        return False

    def calculate_match_score(self, piece1: str, side1: str, piece2: str, side2: str) -> MatchScore:
        """Calcule le score de matching entre deux côtés"""
        # Vérifier le cache
        cache_key = f"{piece1}_{side1}_{piece2}_{side2}"
        if cache_key in self.score_cache:
            return self.score_cache[cache_key]

        # Récupérer les données
        data1 = self.get_piece_data(piece1)
        data2 = self.get_piece_data(piece2)

        if not data1 or not data2:
            return MatchScore(piece1, side1, piece2, side2, 0, 0, 0, False)

        side_data1 = data1['sides'][side1]
        side_data2 = data2['sides'][side2]

        # Vérifier la compatibilité des genres
        if not self.check_gender_compatibility(side_data1['gender'], side_data2['gender']):
            score = MatchScore(piece1, side1, piece2, side2, 0, 0, 0, False)
            self.score_cache[cache_key] = score
            return score

        # Calculer la similarité des couleurs
        color_score = self.calculate_color_similarity(
            side_data1.get('segments', []),
            side_data2.get('segments', [])
        )

        # Calculer la similarité des formes (utiliser l'aire sous la courbe)
        shape_score = 0.5  # Score par défaut
        if 'aire_sous_courbe' in side_data1 and 'aire_sous_courbe' in side_data2:
            area1 = side_data1['aire_sous_courbe']
            area2 = side_data2['aire_sous_courbe']

            # Pour un bon match, les aires doivent être opposées
            if (area1 > 0 and area2 < 0) or (area1 < 0 and area2 > 0):
                # Calculer la similarité basée sur la différence des valeurs absolues
                diff = abs(abs(area1) - abs(area2))
                max_area = max(abs(area1), abs(area2))
                if max_area > 0:
                    shape_score = 1.0 - (diff / (2 * max_area))
                    shape_score = max(0.0, min(1.0, shape_score))
            else:
                shape_score = 0.1  # Pénalité si les signes sont identiques

        # Score total pondéré
        total_score = (self.COLOR_WEIGHT * color_score +
                       self.SHAPE_WEIGHT * shape_score)

        score = MatchScore(piece1, side1, piece2, side2,
                           color_score, shape_score, total_score, True)

        # Mettre en cache
        self.score_cache[cache_key] = score

        return score

    def find_best_piece_for_position(self, row: int, col: int,
                                     available_pieces: List[str]) -> Optional[Tuple[str, int, float]]:
        """
        Trouve la meilleure pièce pour une position donnée
        Assure que les côtés neutres pointent vers l'extérieur
        """
        best_score = -1
        best_piece = None
        best_rotation = 0

        # Déterminer les contraintes de position
        is_corner = (row in [0, self.grid_rows - 1]) and (col in [0, self.grid_cols - 1])
        is_edge = (row == 0 or row == self.grid_rows - 1 or
                   col == 0 or col == self.grid_cols - 1)

        for piece_name in available_pieces:
            piece_data = self.get_piece_data(piece_name)
            if not piece_data:
                continue

            # Trouver la rotation valide pour cette position
            valid_rotation = self.find_valid_rotation(piece_name, row, col)
            if valid_rotation is None:
                continue

            # Calculer le score avec les voisins
            total_score = 0
            match_count = 0

            # Vérifier la compatibilité avec les pièces adjacentes
            # Gauche
            if col > 0 and self.grid[row][col - 1]:
                neighbor = self.grid[row][col - 1]
                neighbor_right = self.get_rotated_side('droite', neighbor.rotation)
                my_left = self.get_rotated_side('gauche', valid_rotation)

                score = self.calculate_match_score(
                    piece_name, my_left,
                    neighbor.filename, neighbor_right
                )

                if not score.is_valid:
                    continue  # Passer à la pièce suivante si incompatible

                total_score += score.total_score
                match_count += 1

            # Haut
            if row > 0 and self.grid[row - 1][col]:
                neighbor = self.grid[row - 1][col]
                neighbor_bottom = self.get_rotated_side('bas', neighbor.rotation)
                my_top = self.get_rotated_side('haut', valid_rotation)

                score = self.calculate_match_score(
                    piece_name, my_top,
                    neighbor.filename, neighbor_bottom
                )

                if not score.is_valid:
                    continue  # Passer à la pièce suivante si incompatible

                total_score += score.total_score
                match_count += 1

            # Si aucun voisin, mais position valide, donner un score de base
            if match_count == 0:
                total_score = 0.5  # Score de base pour les positions sans voisins
                match_count = 1

            avg_score = total_score / match_count
            if avg_score > best_score:
                best_score = avg_score
                best_piece = piece_name
                best_rotation = valid_rotation

        if best_piece:
            return (best_piece, best_rotation, best_score)

        return None

    def place_corner(self, row: int, col: int) -> bool:
        """Place une pièce de coin à une position donnée"""
        for piece_name in self.corner_pieces:
            if piece_name in self.placed_pieces:
                continue

            valid_rotation = self.find_valid_rotation(piece_name, row, col)
            if valid_rotation is not None:
                self.grid[row][col] = PieceInfo(piece_name, valid_rotation, (row, col))
                self.placed_pieces.add(piece_name)
                return True

        return False

    def solve_greedy(self, max_time: int = 120) -> bool:
        """
        Résout le puzzle avec une approche greedy
        """
        start_time = time.time()

        print(f"\nDémarrage de la résolution du puzzle {self.grid_rows}x{self.grid_cols}")
        print(f"Nombre total de pièces: {self.total_pieces}")

        # Étape 1: Placer les coins
        corner_positions = [
            (0, 0), (0, self.grid_cols - 1),
            (self.grid_rows - 1, 0), (self.grid_rows - 1, self.grid_cols - 1)
        ]

        corners_placed = 0
        for row, col in corner_positions:
            if self.place_corner(row, col):
                corners_placed += 1
                piece = self.grid[row][col]
                print(f"Coin placé: {piece.filename} à ({row},{col}) rotation {piece.rotation}°")

        print(f"Coins placés: {corners_placed}/4")

        if corners_placed != 4:
            print("ERREUR: Impossible de placer tous les coins!")
            return False

        # Étape 2: Placer les bordures
        print("\nPlacement des pièces de bordure...")

        # Positions de bordure (dans l'ordre pour favoriser la continuité)
        border_positions = []

        # Bordure du haut (de gauche à droite)
        for col in range(1, self.grid_cols - 1):
            border_positions.append((0, col))

        # Bordure droite (de haut en bas)
        for row in range(1, self.grid_rows - 1):
            border_positions.append((row, self.grid_cols - 1))

        # Bordure du bas (de droite à gauche)
        for col in range(self.grid_cols - 2, 0, -1):
            border_positions.append((self.grid_rows - 1, col))

        # Bordure gauche (de bas en haut)
        for row in range(self.grid_rows - 2, 0, -1):
            border_positions.append((row, 0))

        print(f"Positions de bordure à remplir: {len(border_positions)}")

        # Placer les bordures
        borders_placed = 0
        for row, col in border_positions:
            if time.time() - start_time > max_time:
                print("Temps limite atteint!")
                return False

            candidates = [p for p in self.border_pieces if p not in self.placed_pieces]

            if candidates:
                result = self.find_best_piece_for_position(row, col, candidates)
                if result:
                    piece_name, rotation, score = result
                    self.grid[row][col] = PieceInfo(piece_name, rotation, (row, col))
                    self.placed_pieces.add(piece_name)
                    borders_placed += 1
                    print(f"  Bordure placée: {piece_name} à ({row},{col}) rotation {rotation}° score {score:.3f}")

        print(f"Bordures placées: {borders_placed}/{len(border_positions)}")

        # Étape 3: Remplir l'intérieur
        print("\nRemplissage de l'intérieur...")

        # Remplir ligne par ligne pour maximiser les contraintes
        interior_placed = 0
        for row in range(1, self.grid_rows - 1):
            for col in range(1, self.grid_cols - 1):
                if time.time() - start_time > max_time:
                    print("Temps limite atteint!")
                    return False

                if self.grid[row][col] is None:
                    candidates = [p for p in self.interior_pieces if p not in self.placed_pieces]

                    if candidates:
                        result = self.find_best_piece_for_position(row, col, candidates)
                        if result:
                            piece_name, rotation, score = result
                            self.grid[row][col] = PieceInfo(piece_name, rotation, (row, col))
                            self.placed_pieces.add(piece_name)
                            interior_placed += 1
                            print(
                                f"  Intérieur placé: {piece_name} à ({row},{col}) rotation {rotation}° score {score:.3f}")

        print(f"Pièces intérieures placées: {interior_placed}")

        elapsed_time = time.time() - start_time
        print(f"\nRésolution terminée en {elapsed_time:.2f} secondes")
        print(f"Pièces placées: {len(self.placed_pieces)}/{self.total_pieces}")

        return len(self.placed_pieces) == self.total_pieces

    def verify_solution(self) -> bool:
        """Vérifie que la solution est valide (tous les côtés neutres vers l'extérieur)"""
        print("\n=== VÉRIFICATION DE LA SOLUTION ===")
        errors = []

        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                piece = self.grid[row][col]
                if not piece:
                    continue

                piece_data = self.get_piece_data(piece.filename)
                if not piece_data:
                    continue

                # Vérifier les bordures
                if row == 0:  # Bordure du haut
                    side = self.get_rotated_side('haut', piece.rotation)
                    if piece_data['sides'][side]['gender'] != 'neutre':
                        errors.append(f"Erreur: {piece.filename} à ({row},{col}) - côté haut non neutre")

                if row == self.grid_rows - 1:  # Bordure du bas
                    side = self.get_rotated_side('bas', piece.rotation)
                    if piece_data['sides'][side]['gender'] != 'neutre':
                        errors.append(f"Erreur: {piece.filename} à ({row},{col}) - côté bas non neutre")

                if col == 0:  # Bordure gauche
                    side = self.get_rotated_side('gauche', piece.rotation)
                    if piece_data['sides'][side]['gender'] != 'neutre':
                        errors.append(f"Erreur: {piece.filename} à ({row},{col}) - côté gauche non neutre")

                if col == self.grid_cols - 1:  # Bordure droite
                    side = self.get_rotated_side('droite', piece.rotation)
                    if piece_data['sides'][side]['gender'] != 'neutre':
                        errors.append(f"Erreur: {piece.filename} à ({row},{col}) - côté droite non neutre")

        if errors:
            print("Erreurs trouvées:")
            for error in errors:
                print(f"  - {error}")
            return False
        else:
            print("✓ Solution valide: tous les côtés neutres pointent vers l'extérieur")
            return True

    def get_solution(self) -> Dict:
        """Retourne la solution sous forme de dictionnaire"""
        solution = {
            'grid_size': (self.grid_rows, self.grid_cols),
            'pieces': []
        }

        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                if self.grid[row][col]:
                    piece = self.grid[row][col]
                    solution['pieces'].append({
                        'filename': piece.filename,
                        'position': {'row': row, 'col': col},
                        'rotation': piece.rotation
                    })

        return solution

    def save_solution(self, output_path: str = "puzzle_solution.json"):
        """Sauvegarde la solution dans un fichier JSON"""
        solution = self.get_solution()

        with open(output_path, 'w') as f:
            json.dump(solution, f, indent=2)

        print(f"Solution sauvegardée dans {output_path}")

    def display_solution(self):
        """Affiche la solution sous forme de grille textuelle"""
        print("\n=== SOLUTION DU PUZZLE ===")
        print(f"Grille {self.grid_rows}x{self.grid_cols}\n")

        for row in range(self.grid_rows):
            row_str = ""
            for col in range(self.grid_cols):
                if self.grid[row][col]:
                    piece = self.grid[row][col]
                    # Extraire le numéro de la pièce
                    piece_num = piece.filename.replace('piece_', '').replace('.png', '')
                    row_str += f"{piece_num:>3}°{piece.rotation:<3} "
                else:
                    row_str += "  ---   "
            print(row_str)

        print("\nLégende: numéro°rotation")


def main():
    """Fonction principale pour tester le matcher"""
    # Créer le matcher
    matcher = PuzzleMatcher(
        puzzle_data_path="puzzle_complete_data.json",
        grid_size=(4, 6)  # 24 pièces
    )

    # Résoudre le puzzle avec l'approche greedy
    success = matcher.solve_greedy(max_time=120)  # 2 minutes max

    if success:
        print("\n✓ Puzzle résolu avec succès!")

        # Vérifier la solution
        is_valid = matcher.verify_solution()

        # Afficher la solution
        matcher.display_solution()

        # Sauvegarder la solution
        matcher.save_solution("puzzle_solution.json")

        # Afficher quelques statistiques
        solution = matcher.get_solution()
        print(f"\nNombre de pièces placées: {len(solution['pieces'])}")

        # Créer un rapport détaillé
        print("\n=== RAPPORT DE MATCHING ===")
        total_score = 0
        match_count = 0

        for piece_info in solution['pieces']:
            row = piece_info['position']['row']
            col = piece_info['position']['col']
            piece_name = piece_info['filename']
            rotation = piece_info['rotation']

            # Calculer les scores avec les voisins
            neighbors = [
                (row - 1, col, 'haut', 'bas'),
                (row, col + 1, 'droite', 'gauche'),
                (row + 1, col, 'bas', 'haut'),
                (row, col - 1, 'gauche', 'droite')
            ]

            for n_row, n_col, my_side, neighbor_side in neighbors:
                if 0 <= n_row < matcher.grid_rows and 0 <= n_col < matcher.grid_cols:
                    neighbor = matcher.grid[n_row][n_col]
                    if neighbor:
                        my_rotated = matcher.get_rotated_side(my_side, rotation)
                        neighbor_rotated = matcher.get_rotated_side(neighbor_side, neighbor.rotation)

                        score = matcher.calculate_match_score(
                            piece_name, my_rotated,
                            neighbor.filename, neighbor_rotated
                        )

                        if score.is_valid:
                            total_score += score.total_score
                            match_count += 1

        if match_count > 0:
            avg_score = total_score / match_count
            print(f"Score moyen des connexions: {avg_score:.3f}")
            print(f"Nombre total de connexions: {match_count}")

    else:
        print("\n✗ Impossible de résoudre complètement le puzzle")
        print(f"Pièces placées: {len(matcher.placed_pieces)}/{matcher.total_pieces}")

        # Afficher quand même la solution partielle
        matcher.display_solution()
        matcher.save_solution("puzzle_solution_partial.json")


if __name__ == "__main__":
    main()