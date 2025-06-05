import numpy as np
import json
import time
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.spatial.distance import euclidean
import cv2
import math
import matplotlib.pyplot as plt
import os
from PIL import Image

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


class DFSPuzzleSolver:
    def __init__(self, puzzle_data_path: str):
        """
        Initialise le solveur de puzzle avec DFS et backtracking

        Args:
            puzzle_data_path: Chemin vers le fichier JSON contenant toutes les données
        """
        # Charger les données
        with open(puzzle_data_path, 'r') as f:
            self.puzzle_data = json.load(f)

        print(f"Données chargées pour {len(self.puzzle_data)} pièces")

        # Déterminer automatiquement la taille de la grille
        self.total_pieces = len(self.puzzle_data)
        self.grid_rows, self.grid_cols = self._determine_grid_size()
        
        print(f"Taille de grille déterminée: {self.grid_rows}x{self.grid_cols}")

        # Poids pour le scoring
        self.COLOR_WEIGHT = 0.7
        self.SHAPE_WEIGHT = 0.3

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

        # Variables pour le DFS
        self.max_backtrack_depth = 5
        self.backtrack_count = 0
        self.solution_found = False

    def _determine_grid_size(self) -> Tuple[int, int]:
        """Détermine automatiquement la taille de la grille basée sur le nombre de pièces"""
        n = self.total_pieces
        
        # Trouver les facteurs possibles
        factors = []
        for i in range(1, int(math.sqrt(n)) + 1):
            if n % i == 0:
                factors.append((i, n // i))
        
        # Préférer les grilles proches du carré
        best_ratio = float('inf')
        best_size = factors[0]
        
        for rows, cols in factors:
            ratio = max(rows/cols, cols/rows)
            if ratio < best_ratio:
                best_ratio = ratio
                best_size = (rows, cols)
        
        return best_size

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

        return max(0.0, min(1.0, similarity))

    def calculate_shape_similarity(self, points1: List, points2: List) -> float:
        """
        Calcule la similarité entre deux formes normalisées
        """
        if not points1 or not points2 or len(points1) != 17 or len(points2) != 17:
            return 0.5  # Score neutre par défaut

        try:
            # Convertir en numpy arrays
            pts1 = np.array(points1)
            pts2 = np.array(points2)

            # Inverser pts2 pour le matching
            pts2_reversed = pts2[::-1]

            # Calculer la différence des y (hauteurs)
            y_diff = 0
            for i in range(len(pts1)):
                y_diff += abs(pts1[i][1] + pts2_reversed[i][1])  # Somme doit être proche de 0

            # Normaliser
            avg_y_diff = y_diff / len(pts1)
            
            # Plus la différence est proche de 0, meilleur est le score
            similarity = 1.0 / (1.0 + avg_y_diff / 10.0)

            return max(0.0, min(1.0, similarity))

        except Exception as e:
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

        # Calculer la similarité des formes
        shape_score = self.calculate_shape_similarity(
            side_data1.get('points_normalises', []),
            side_data2.get('points_normalises', [])
        )

        # Score total pondéré
        total_score = (self.COLOR_WEIGHT * color_score +
                       self.SHAPE_WEIGHT * shape_score)

        score = MatchScore(piece1, side1, piece2, side2,
                           color_score, shape_score, total_score, True)

        # Mettre en cache
        self.score_cache[cache_key] = score

        return score

    def is_valid_placement(self, piece_name: str, rotation: int, row: int, col: int) -> bool:
        """Vérifie si une pièce peut être placée à une position donnée"""
        piece_data = self.get_piece_data(piece_name)
        if not piece_data:
            return False

        # Vérifier les contraintes de bordure
        if row == 0:  # Bordure haut
            top_side = self.get_rotated_side('haut', rotation)
            if piece_data['sides'][top_side]['gender'] != 'neutre':
                return False
        else:
            top_side = self.get_rotated_side('haut', rotation)
            if piece_data['sides'][top_side]['gender'] == 'neutre':
                return False

        if row == self.grid_rows - 1:  # Bordure bas
            bottom_side = self.get_rotated_side('bas', rotation)
            if piece_data['sides'][bottom_side]['gender'] != 'neutre':
                return False
        else:
            bottom_side = self.get_rotated_side('bas', rotation)
            if piece_data['sides'][bottom_side]['gender'] == 'neutre':
                return False

        if col == 0:  # Bordure gauche
            left_side = self.get_rotated_side('gauche', rotation)
            if piece_data['sides'][left_side]['gender'] != 'neutre':
                return False
        else:
            left_side = self.get_rotated_side('gauche', rotation)
            if piece_data['sides'][left_side]['gender'] == 'neutre':
                return False

        if col == self.grid_cols - 1:  # Bordure droite
            right_side = self.get_rotated_side('droite', rotation)
            if piece_data['sides'][right_side]['gender'] != 'neutre':
                return False
        else:
            right_side = self.get_rotated_side('droite', rotation)
            if piece_data['sides'][right_side]['gender'] == 'neutre':
                return False

        return True

    def get_compatibility_score(self, piece_name: str, rotation: int, row: int, col: int) -> float:
        """Calcule le score de compatibilité d'une pièce à une position"""
        if not self.is_valid_placement(piece_name, rotation, row, col):
            return -1

        total_score = 0
        neighbor_count = 0

        # Vérifier chaque voisin
        neighbors = [
            (row - 1, col, 'haut', 'bas'),    # Voisin du haut
            (row, col + 1, 'droite', 'gauche'), # Voisin de droite
            (row + 1, col, 'bas', 'haut'),    # Voisin du bas
            (row, col - 1, 'gauche', 'droite') # Voisin de gauche
        ]

        for n_row, n_col, my_side, neighbor_side in neighbors:
            if (0 <= n_row < self.grid_rows and 0 <= n_col < self.grid_cols and
                self.grid[n_row][n_col] is not None):
                
                neighbor = self.grid[n_row][n_col]
                my_rotated_side = self.get_rotated_side(my_side, rotation)
                neighbor_rotated_side = self.get_rotated_side(neighbor_side, neighbor.rotation)

                match_score = self.calculate_match_score(
                    piece_name, my_rotated_side,
                    neighbor.filename, neighbor_rotated_side
                )

                if not match_score.is_valid:
                    return -1  # Incompatible

                total_score += match_score.total_score
                neighbor_count += 1

        # Si pas de voisins, retourner un score neutre
        if neighbor_count == 0:
            return 0.5

        return total_score / neighbor_count

    def get_next_position(self) -> Optional[Tuple[int, int]]:
        """Retourne la prochaine position vide dans la grille"""
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                if self.grid[row][col] is None:
                    return (row, col)
        return None

    def get_best_candidates(self, row: int, col: int, available_pieces: List[str], max_candidates: int = 5) -> List[Tuple[str, int, float]]:
        """Retourne les meilleurs candidats pour une position triés par score"""
        candidates = []

        for piece_name in available_pieces:
            for rotation in [0, 90, 180, 270]:
                score = self.get_compatibility_score(piece_name, rotation, row, col)
                if score >= 0:  # Valide
                    candidates.append((piece_name, rotation, score))

        # Trier par score décroissant
        candidates.sort(key=lambda x: x[2], reverse=True)
        
        # Retourner les meilleurs candidats
        return candidates[:max_candidates]

    def place_piece(self, piece_name: str, rotation: int, row: int, col: int):
        """Place une pièce sur la grille"""
        self.grid[row][col] = PieceInfo(piece_name, rotation, (row, col))
        self.placed_pieces.add(piece_name)

    def remove_piece(self, row: int, col: int):
        """Retire une pièce de la grille"""
        if self.grid[row][col]:
            piece_name = self.grid[row][col].filename
            self.placed_pieces.remove(piece_name)
            self.grid[row][col] = None

    def solve_dfs(self, max_time: int = 300) -> bool:
        """
        Résout le puzzle avec DFS et backtracking

        Args:
            max_time: Temps maximum en secondes (default: 300s = 5 minutes)

        Returns:
            True si résolu, False sinon
        """
        start_time = time.time()
        self.solution_found = False
        self.backtrack_count = 0

        print(f"\nDémarrage de la résolution DFS du puzzle {self.grid_rows}x{self.grid_cols}")
        print(f"Nombre total de pièces: {self.total_pieces}")

        # Étape 1: Choisir une pièce de coin aléatoire pour commencer
        if self.corner_pieces:
            start_piece = random.choice(self.corner_pieces)
            print(f"Pièce de coin sélectionnée: {start_piece}")
        else:
            start_piece = random.choice([p['filename'] for p in self.puzzle_data])
            print(f"Aucun coin disponible, pièce aléatoire sélectionnée: {start_piece}")

        # Placer la pièce de départ au coin supérieur gauche
        piece_data = self.get_piece_data(start_piece)
        if piece_data:
            # Trouver la bonne rotation pour le coin supérieur gauche
            for rotation in [0, 90, 180, 270]:
                if self.is_valid_placement(start_piece, rotation, 0, 0):
                    self.place_piece(start_piece, rotation, 0, 0)
                    print(f"Pièce de départ placée: {start_piece} à (0,0) avec rotation {rotation}")
                    break

            # Commencer le DFS
            available_pieces = [p['filename'] for p in self.puzzle_data if p['filename'] not in self.placed_pieces]
            
            success = self._dfs_recursive(available_pieces, start_time, max_time)
            
            elapsed_time = time.time() - start_time
            print(f"\nRésolution terminée en {elapsed_time:.2f} secondes")
            print(f"Nombre de backtracks: {self.backtrack_count}")
            print(f"Pièces placées: {len(self.placed_pieces)}/{self.total_pieces}")

            return success

        return False

    def _dfs_recursive(self, available_pieces: List[str], start_time: float, max_time: int) -> bool:
        """Fonction récursive DFS avec backtracking"""
        # Vérifier le temps limite
        if time.time() - start_time > max_time:
            print("Temps limite atteint!")
            return False

        # Si toutes les pièces sont placées, succès!
        if len(self.placed_pieces) == self.total_pieces:
            self.solution_found = True
            return True

        # Trouver la prochaine position vide
        next_pos = self.get_next_position()
        if not next_pos:
            return True  # Plus de positions vides

        row, col = next_pos

        # Obtenir les meilleurs candidats pour cette position
        candidates = self.get_best_candidates(row, col, available_pieces)

        if not candidates:
            # Aucun candidat valide, backtrack
            self.backtrack_count += 1
            return False

        # Essayer chaque candidat
        for piece_name, rotation, score in candidates:
            # Placer la pièce
            self.place_piece(piece_name, rotation, row, col)
            
            # Mettre à jour les pièces disponibles
            new_available = [p for p in available_pieces if p != piece_name]
            
            # Appel récursif
            if self._dfs_recursive(new_available, start_time, max_time):
                return True
            
            # Backtrack: retirer la pièce
            self.remove_piece(row, col)

        # Aucune solution trouvée avec les candidats actuels
        self.backtrack_count += 1
        return False

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

    def save_solution(self, output_path: str = "puzzle_solution_dfs.json"):
        """Sauvegarde la solution dans un fichier JSON"""
        solution = self.get_solution()

        with open(output_path, 'w') as f:
            json.dump(solution, f, indent=2)

        print(f"Solution sauvegardée dans {output_path}")

    def display_solution(self):
        """Affiche la solution sous forme de grille textuelle"""
        print("\n=== SOLUTION DU PUZZLE (DFS) ===")
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

        # Afficher les pièces non placées
        if len(self.placed_pieces) < self.total_pieces:
            unplaced = []
            for piece in self.puzzle_data:
                if piece['filename'] not in self.placed_pieces:
                    unplaced.append(piece['filename'])

            print(f"\nPièces non placées ({len(unplaced)}):")
            for piece in unplaced:
                print(f"  - {piece}")


def main():
    """Fonction principale pour tester le solveur DFS"""
    # Créer le solveur DFS
    solver = DFSPuzzleSolver(
        puzzle_data_path="puzzle_complete_data.json"
    )

    # Résoudre le puzzle avec DFS et backtracking
    success = solver.solve_dfs(max_time=300)  # 5 minutes max

    if success:
        print("\n✓ Puzzle résolu avec succès!")

        # Afficher la solution
        solver.display_solution()

        # Sauvegarder la solution
        solver.save_solution("puzzle_solution_dfs.json")

        # Afficher quelques statistiques
        solution = solver.get_solution()
        print(f"\nNombre de pièces placées: {len(solution['pieces'])}")
        print(f"Nombre de backtracks effectués: {solver.backtrack_count}")

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
                if 0 <= n_row < solver.grid_rows and 0 <= n_col < solver.grid_cols:
                    neighbor = solver.grid[n_row][n_col]
                    if neighbor:
                        my_rotated = solver.get_rotated_side(my_side, rotation)
                        neighbor_rotated = solver.get_rotated_side(neighbor_side, neighbor.rotation)

                        score = solver.calculate_match_score(
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
        print(f"Pièces placées: {len(solver.placed_pieces)}/{solver.total_pieces}")

        # Afficher quand même la solution partielle
        solver.display_solution()
        solver.save_solution("puzzle_solution_dfs_partial.json")

if __name__ == "__main__":
    main()