import numpy as np
import json
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.spatial.distance import euclidean
import cv2


@dataclass
class PieceInfo:
    """Information sur une pièce de puzzle placée"""
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
        self.COLOR_WEIGHT = 1.0
        self.SHAPE_WEIGHT = 0.0

        # Mappings des côtés selon la rotation
        self.side_rotation_map = {
            0: {'haut': 'haut', 'droite': 'droite', 'bas': 'bas', 'gauche': 'gauche'},
            90: {'haut': 'gauche', 'droite': 'haut', 'bas': 'droite', 'gauche': 'bas'},
            180: {'haut': 'bas', 'droite': 'gauche', 'bas': 'haut', 'gauche': 'droite'},
            270: {'haut': 'droite', 'droite': 'bas', 'bas': 'gauche', 'gauche': 'haut'}
        }

        # Grille de solution (initialement vide)
        self.grid: List[List[Optional[PieceInfo]]] = [
            [None for _ in range(self.grid_cols)] for _ in range(self.grid_rows)
        ]
        self.placed_pieces = set()

        # Cache des scores pour optimisation
        self.score_cache: Dict[str, MatchScore] = {}

        # Classifier les pièces par type (coins, bordures, intérieures)
        self.classify_pieces()

    def classify_pieces(self):
        """Classifie les pièces en coins, bordures et intérieures"""
        self.corner_pieces: List[str] = []
        self.border_pieces: List[str] = []
        self.interior_pieces: List[str] = []

        for piece in self.puzzle_data:
            neutral_count = sum(
                1 for side in ['haut', 'droite', 'bas', 'gauche']
                if piece['sides'][side]['gender'] == 'neutre'
            )

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

    def get_piece_data(self, piece_name: str) -> Optional[Dict]:
        """Récupère les données d'une pièce à partir de son filename"""
        for piece in self.puzzle_data:
            if piece['filename'] == piece_name:
                return piece
        return None

    def get_rotated_side(self, side: str, rotation: int) -> str:
        """Retourne le nom du côté après application de la rotation (0, 90, 180, 270)"""
        return self.side_rotation_map[rotation][side]

    def get_neutral_sides(self, piece_data: Dict) -> List[str]:
        """Retourne la liste des côtés neutres d'une pièce (genre == 'neutre')"""
        return [
            side for side in ['haut', 'droite', 'bas', 'gauche']
            if piece_data['sides'][side]['gender'] == 'neutre'
        ]

    def calculate_color_similarity(self, colors1: List[Dict], colors2: List[Dict]) -> float:
        """
        Calcule la similarité entre deux séries de couleurs (listes de segments).
        On inverse colors2 pour matcher le bon sens.
        Retourne un score entre 0.0 et 1.0 (1.0 = identique).
        """
        if not colors1 or not colors2:
            return 0.0

        total_distance = 0.0
        count = 0

        # Inverser la seconde liste pour le matching "face-à-face"
        colors2_reversed = list(reversed(colors2))

        # On compare segment par segment
        min_segments = min(len(colors1), len(colors2_reversed))
        for i in range(min_segments):
            color1 = np.array(colors1[i]['color_rgb'], dtype=float)
            color2 = np.array(colors2_reversed[i]['color_rgb'], dtype=float)
            distance = euclidean(color1, color2)
            total_distance += distance
            count += 1

        if count == 0:
            return 0.0

        avg_distance = total_distance / count
        max_distance = 255 * np.sqrt(3)  # distance max possible en RGB

        # Similarité normalisée
        similarity = 1.0 - (avg_distance / max_distance)
        return max(0.0, min(1.0, similarity))

    def calculate_shape_similarity(self, points1: List, points2: List) -> float:
        """
        Calcule la similarité entre deux formes normalisées (listes de 17 points).
        Version simplifiée basée sur l'aire sous la courbe.
        Retourne un score entre 0.0 et 1.0, ou 0.5 par défaut si données manquantes.
        """
        if (not points1 or not points2) or (len(points1) != 17 or len(points2) != 17):
            return 0.5  # Score neutre si pas de données correctes

        try:
            pts1 = np.array(points1, dtype=float)
            pts2 = np.array(points2, dtype=float)
            pts2_reversed = pts2[::-1]

            area1 = float(pts1[-1][1])
            area2 = float(pts2_reversed[-1][1])

            # Pour un bon match, les aires doivent avoir des signes opposés
            if (area1 > 0 and area2 < 0) or (area1 < 0 and area2 > 0):
                diff = abs(abs(area1) - abs(area2))
                max_area = max(abs(area1), abs(area2))
                if max_area > 0:
                    similarity = 1.0 - (diff / max_area)
                else:
                    similarity = 0.5
            else:
                similarity = 0.1  # Faible si signes identiques

            return max(0.0, min(1.0, similarity))
        except Exception:
            return 0.5

    def check_gender_compatibility(self, gender1: str, gender2: str) -> bool:
        """Vérifie si deux côtés sont compatibles en genre (male <-> femelle)"""
        if (gender1 == 'male' and gender2 == 'femelle') or (gender1 == 'femelle' and gender2 == 'male'):
            return True
        return False

    def calculate_match_score(self, piece1: str, side1: str, piece2: str, side2: str) -> MatchScore:
        """
        Calcule le score de matching entre le côté `side1` de `piece1`
        et le côté `side2` de `piece2`.
        Retourne un objet MatchScore.
        """
        # Vérifier le cache
        cache_key = f"{piece1}_{side1}_{piece2}_{side2}"
        if cache_key in self.score_cache:
            return self.score_cache[cache_key]

        data1 = self.get_piece_data(piece1)
        data2 = self.get_piece_data(piece2)
        if not data1 or not data2:
            score = MatchScore(piece1, side1, piece2, side2, 0.0, 0.0, 0.0, False)
            self.score_cache[cache_key] = score
            return score

        side_data1 = data1['sides'][side1]
        side_data2 = data2['sides'][side2]

        # Vérifier la compatibilité des genres
        if not self.check_gender_compatibility(side_data1['gender'], side_data2['gender']):
            score = MatchScore(piece1, side1, piece2, side2, 0.0, 0.0, 0.0, False)
            self.score_cache[cache_key] = score
            return score

        # Calcul de la similarité des couleurs
        color_score = self.calculate_color_similarity(
            side_data1.get('segments', []),
            side_data2.get('segments', [])
        )

        # Calcul de la similarité des formes (aire sous la courbe)
        shape_score = 0.5  # Score par défaut
        if 'aire_sous_courbe' in side_data1 and 'aire_sous_courbe' in side_data2:
            area1 = side_data1['aire_sous_courbe']
            area2 = side_data2['aire_sous_courbe']
            if (area1 > 0 and area2 < 0) or (area1 < 0 and area2 > 0):
                diff = abs(abs(area1) - abs(area2))
                max_area = max(abs(area1), abs(area2))
                if max_area > 0:
                    shape_score = 1.0 - (diff / (2 * max_area))
                    shape_score = max(0.0, min(1.0, shape_score))
            else:
                shape_score = 0.1

        # Score total pondéré
        total_score = (self.COLOR_WEIGHT * color_score +
                       self.SHAPE_WEIGHT * shape_score)

        score = MatchScore(
            piece1, side1, piece2, side2,
            color_score, shape_score, total_score, True
        )
        self.score_cache[cache_key] = score
        return score

    def find_best_piece_for_position(self, row: int, col: int,
                                     available_pieces: List[str]) -> Optional[Tuple[str, int, float]]:
        """
        Trouve la meilleure pièce (et orientation) pour la position (row, col),
        en comparant aux voisins déjà placés. Retourne (piece_name, rotation, score)
        ou None si aucun candidat valide.
        """
        best_score = -1.0
        best_piece = None
        best_rotation = 0

        # Détecter si la position est sur un bord
        is_top_edge = (row == 0)
        is_bottom_edge = (row == self.grid_rows - 1)
        is_left_edge = (col == 0)
        is_right_edge = (col == self.grid_cols - 1)

        for piece_name in available_pieces:
            piece_data = self.get_piece_data(piece_name)
            if piece_data is None:
                continue

            # Tester les 4 rotations possibles
            for rotation in [0, 90, 180, 270]:
                valid = True
                total_score = 0.0
                match_count = 0

                # Vérifier que la pièce respecte les contraintes de bord
                # Haut
                top_side = self.get_rotated_side('haut', rotation)
                if is_top_edge:
                    if piece_data['sides'][top_side]['gender'] != 'neutre':
                        continue
                else:
                    if piece_data['sides'][top_side]['gender'] == 'neutre':
                        continue
                # Bas
                bottom_side = self.get_rotated_side('bas', rotation)
                if is_bottom_edge:
                    if piece_data['sides'][bottom_side]['gender'] != 'neutre':
                        continue
                else:
                    if piece_data['sides'][bottom_side]['gender'] == 'neutre':
                        continue
                # Gauche
                left_side = self.get_rotated_side('gauche', rotation)
                if is_left_edge:
                    if piece_data['sides'][left_side]['gender'] != 'neutre':
                        continue
                else:
                    if piece_data['sides'][left_side]['gender'] == 'neutre':
                        continue
                # Droite
                right_side = self.get_rotated_side('droite', rotation)
                if is_right_edge:
                    if piece_data['sides'][right_side]['gender'] != 'neutre':
                        continue
                else:
                    if piece_data['sides'][right_side]['gender'] == 'neutre':
                        continue

                # Vérifier la compatibilité avec le voisin de gauche
                if col > 0 and self.grid[row][col - 1] is not None:
                    neighbor = self.grid[row][col - 1]
                    neighbor_right = self.get_rotated_side('droite', neighbor.rotation)
                    my_left = self.get_rotated_side('gauche', rotation)
                    score = self.calculate_match_score(
                        piece_name, my_left,
                        neighbor.filename, neighbor_right
                    )
                    if not score.is_valid:
                        valid = False
                    else:
                        total_score += score.total_score
                        match_count += 1
                    if not valid:
                        continue

                # Vérifier la compatibilité avec le voisin du haut
                if valid and row > 0 and self.grid[row - 1][col] is not None:
                    neighbor = self.grid[row - 1][col]
                    neighbor_bottom = self.get_rotated_side('bas', neighbor.rotation)
                    my_top = self.get_rotated_side('haut', rotation)
                    score = self.calculate_match_score(
                        piece_name, my_top,
                        neighbor.filename, neighbor_bottom
                    )
                    if not score.is_valid:
                        valid = False
                    else:
                        total_score += score.total_score
                        match_count += 1
                    if not valid:
                        continue

                # Si aucun voisin et valide, donner un score de base (50% chance)
                if valid and match_count == 0:
                    avg_score = 0.5
                elif valid:
                    avg_score = total_score / match_count
                else:
                    continue  # Pièce invalide pour cette position/rotation

                # Mettre à jour le meilleur choix
                if avg_score > best_score:
                    best_score = avg_score
                    best_piece = piece_name
                    best_rotation = rotation

        if best_piece is not None:
            return (best_piece, best_rotation, best_score)
        return None

    def place_corner(self, row: int, col: int) -> bool:
        """
        Place une pièce de coin exactement à la position (row, col).
        Retourne True si succès, False sinon.
        """
        for piece_name in self.corner_pieces:
            if piece_name in self.placed_pieces:
                continue
            piece_data = self.get_piece_data(piece_name)
            if piece_data is None:
                continue

            neutral_sides = self.get_neutral_sides(piece_data)
            for rotation in [0, 90, 180, 270]:
                rotated_neutrals = [
                    self.get_rotated_side(side, rotation) for side in neutral_sides
                ]

                # Construire la liste des "neutres attendus" pour cette position
                expected_neutrals = []
                if row == 0:
                    expected_neutrals.append('haut')
                if row == self.grid_rows - 1:
                    expected_neutrals.append('bas')
                if col == 0:
                    expected_neutrals.append('gauche')
                if col == self.grid_cols - 1:
                    expected_neutrals.append('droite')

                # Vérifier que l'ensemble correspond exactement
                if set(rotated_neutrals) == set(expected_neutrals):
                    self.grid[row][col] = PieceInfo(piece_name, rotation, (row, col))
                    self.placed_pieces.add(piece_name)
                    return True
        return False

    def solve_greedy(self, max_time: int = 120) -> bool:
        """
        Résout le puzzle avec une approche gloutonne (greedy).
        Tente d’abord les coins, puis les bordures, puis l’intérieur.

        Args:
            max_time: Temps limite en secondes (default: 120s)
        Returns:
            True si toutes les pièces ont pu être placées, False sinon.
        """
        start_time = time.time()
        print(f"\nDémarrage de la résolution du puzzle {self.grid_rows}x{self.grid_cols}")
        print(f"Nombre total de pièces : {self.total_pieces}")

        # --- Étape 1 : placer les 4 coins ---
        corner_positions = [
            (0, 0), (0, self.grid_cols - 1),
            (self.grid_rows - 1, 0), (self.grid_rows - 1, self.grid_cols - 1)
        ]
        corners_placed = 0
        for (r, c) in corner_positions:
            if self.place_corner(r, c):
                corners_placed += 1
        print(f"Coins placés : {corners_placed}/4")
        if corners_placed != 4:
            print("ERREUR : Impossible de placer tous les coins !")
            return False

        # --- Étape 2 : placer les bordures ---
        print("\nPlacement des pièces de bordure...")
        border_positions: List[Tuple[int, int]] = []
        # Bordure du haut (cols 1 à cols-2)
        for c in range(1, self.grid_cols - 1):
            border_positions.append((0, c))
        # Bordure droite (rows 1 à rows-2)
        for r in range(1, self.grid_rows - 1):
            border_positions.append((r, self.grid_cols - 1))
        # Bordure bas (cols cols-2 à 1)
        for c in range(self.grid_cols - 2, 0, -1):
            border_positions.append((self.grid_rows - 1, c))
        # Bordure gauche (rows rows-2 à 1)
        for r in range(self.grid_rows - 2, 0, -1):
            border_positions.append((r, 0))

        print(f"Positions de bordure à remplir : {len(border_positions)}")
        borders_placed = 0
        for (r, c) in border_positions:
            if time.time() - start_time > max_time:
                print("Temps limite atteint !")
                return False
            candidates = [p for p in self.border_pieces if p not in self.placed_pieces]
            if not candidates:
                continue
            result = self.find_best_piece_for_position(r, c, candidates)
            if result:
                piece_name, rotation, score = result
                self.grid[r][c] = PieceInfo(piece_name, rotation, (r, c))
                self.placed_pieces.add(piece_name)
                borders_placed += 1
                print(f"  Bordure placée : {piece_name} à ({r},{c}) avec score {score:.3f}")

        print(f"Bordures placées : {borders_placed}/{len(border_positions)}")
        print(f"Total pièces placées : {len(self.placed_pieces)}/{self.total_pieces}")

        # --- Étape 3 : remplir l’intérieur ligne par ligne ---
        print("\nRemplissage de l’intérieur...")
        interior_placed = 0
        for r in range(1, self.grid_rows - 1):
            for c in range(1, self.grid_cols - 1):
                if time.time() - start_time > max_time:
                    print("Temps limite atteint !")
                    return False
                if self.grid[r][c] is not None:
                    continue
                candidates = [p for p in self.interior_pieces if p not in self.placed_pieces]
                if not candidates:
                    continue
                result = self.find_best_piece_for_position(r, c, candidates)
                if result:
                    piece_name, rotation, score = result
                    self.grid[r][c] = PieceInfo(piece_name, rotation, (r, c))
                    self.placed_pieces.add(piece_name)
                    interior_placed += 1
                    print(f"  Intérieur placé : {piece_name} à ({r},{c}) avec score {score:.3f}")

        print(f"Pièces intérieures placées : {interior_placed}")

        # --- Étape 4 : tenter de forcer les pièces restantes ---
        if len(self.placed_pieces) < self.total_pieces:
            print(f"\nTentative de placement des {self.total_pieces - len(self.placed_pieces)} pièces restantes...")
            remaining_pieces = [
                p['filename'] for p in self.puzzle_data
                if p['filename'] not in self.placed_pieces
            ]
            for r in range(self.grid_rows):
                for c in range(self.grid_cols):
                    if self.grid[r][c] is None and remaining_pieces:
                        best_forced = None
                        best_forced_score = -float('inf')
                        for piece_name in remaining_pieces:
                            for rotation in [0, 90, 180, 270]:
                                score = self.calculate_position_score(piece_name, rotation, r, c)
                                if score > best_forced_score:
                                    best_forced_score = score
                                    best_forced = (piece_name, rotation)
                        if best_forced:
                            piece_name, rotation = best_forced
                            self.grid[r][c] = PieceInfo(piece_name, rotation, (r, c))
                            self.placed_pieces.add(piece_name)
                            remaining_pieces.remove(piece_name)
                            print(f"  Placement forcé : {piece_name} à ({r},{c}) (score {best_forced_score:.3f})")

        elapsed_time = time.time() - start_time
        print(f"\nRésolution terminée en {elapsed_time:.2f} secondes")
        print(f"Pièces placées : {len(self.placed_pieces)}/{self.total_pieces}")

        return len(self.placed_pieces) == self.total_pieces

    def calculate_position_score(self, piece_name: str, rotation: int, row: int, col: int) -> float:
        """
        Calcule un score heuristique pour placer `piece_name` tourné de `rotation`
        à la position (row, col), même si le matching n'est pas « parfait ».
        Sert au placement forcé.
        """
        piece_data = self.get_piece_data(piece_name)
        if piece_data is None:
            return -float('inf')

        score = 0.0
        matches = 0
        penalty = 0

        # Pénalités pour violation des contraintes de bord
        if row == 0:
            side = self.get_rotated_side('haut', rotation)
            if piece_data['sides'][side]['gender'] != 'neutre':
                penalty += 10
        elif row == self.grid_rows - 1:
            side = self.get_rotated_side('bas', rotation)
            if piece_data['sides'][side]['gender'] != 'neutre':
                penalty += 10

        if col == 0:
            side = self.get_rotated_side('gauche', rotation)
            if piece_data['sides'][side]['gender'] != 'neutre':
                penalty += 10
        elif col == self.grid_cols - 1:
            side = self.get_rotated_side('droite', rotation)
            if piece_data['sides'][side]['gender'] != 'neutre':
                penalty += 10

        if penalty == 0:
            score += 0.5  # petit bonus si respecte les bords

        # Vérifier la compatibilité avec chaque voisin déjà placé
        neighbors = [
            (row - 1, col, 'haut', 'bas'),
            (row, col + 1, 'droite', 'gauche'),
            (row + 1, col, 'bas', 'haut'),
            (row, col - 1, 'gauche', 'droite'),
        ]
        for (nr, nc, my_side, neighbor_side) in neighbors:
            if 0 <= nr < self.grid_rows and 0 <= nc < self.grid_cols:
                neighbor = self.grid[nr][nc]
                if neighbor:
                    my_rotated_side = self.get_rotated_side(my_side, rotation)
                    neighbor_rotated_side = self.get_rotated_side(neighbor_side, neighbor.rotation)
                    match_score = self.calculate_match_score(
                        piece_name, my_rotated_side,
                        neighbor.filename, neighbor_rotated_side
                    )
                    if match_score.is_valid:
                        score += match_score.total_score
                        matches += 1
                    else:
                        score -= 5  # pénalité forte si incompatibilité de genre

        if matches > 0:
            score = score / matches

        score -= penalty
        return score

    def get_solution(self) -> Dict:
        """Retourne la solution actuelle sous forme d'un dictionnaire JSON-compatible"""
        solution = {
            'grid_size': [self.grid_rows, self.grid_cols],
            'pieces': []
        }
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                if self.grid[r][c] is not None:
                    piece = self.grid[r][c]
                    solution['pieces'].append({
                        'filename': piece.filename,
                        'position': {'row': r, 'col': c},
                        'rotation': piece.rotation
                    })
        return solution

    def save_solution(self, output_path: str = "puzzle_solution.json"):
        """Sauvegarde la solution (ou partielle) dans un fichier JSON"""
        solution = self.get_solution()
        with open(output_path, 'w') as f:
            json.dump(solution, f, indent=2)
        print(f"Solution sauvegardée dans {output_path}")

    def display_solution(self):
        """Affiche la solution sous forme textuelle (numéro°rotation)"""
        print("\n=== SOLUTION DU PUZZLE ===")
        print(f"Grille {self.grid_rows}x{self.grid_cols}\n")
        for r in range(self.grid_rows):
            row_str = ""
            for c in range(self.grid_cols):
                if self.grid[r][c]:
                    piece = self.grid[r][c]
                    # Extraire un « numero » à partir du filename (piece_XX.png → XX)
                    piece_num = piece.filename.replace('piece_', '').replace('.png', '')
                    row_str += f"{piece_num:>3}°{piece.rotation:<3} "
                else:
                    row_str += "  ---   "
            print(row_str)

        print("\nLégende : numéro°rotation")
        # Afficher les pièces non placées (le cas échéant)
        if len(self.placed_pieces) < self.total_pieces:
            unplaced = [
                piece['filename'] for piece in self.puzzle_data
                if piece['filename'] not in self.placed_pieces
            ]
            print(f"\nPièces non placées ({len(unplaced)}) :")
            for p in unplaced:
                print(f"  - {p}")


def main():
    """Fonction principale pour tester le matcher"""
    matcher = PuzzleMatcher(
        puzzle_data_path="puzzle_complete_data.json",
        grid_size=(4, 6)  # 24 pièces
    )

    success = matcher.solve_greedy(max_time=120)  # 2 minutes max

    if success:
        print("\n✓ Puzzle résolu avec succès !")
        matcher.display_solution()
        matcher.save_solution("puzzle_solution.json")
        solution = matcher.get_solution()
        print(f"\nNombre de pièces placées : {len(solution['pieces'])}")
    else:
        print("\n✗ Impossible de résoudre complètement le puzzle")
        print(f"Pièces placées : {len(matcher.placed_pieces)}/{matcher.total_pieces}")
        matcher.display_solution()
        matcher.save_solution("puzzle_solution_partial.json")


if __name__ == "__main__":
    main()
