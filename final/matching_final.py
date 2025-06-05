import json
import numpy as np
from scipy.spatial.distance import euclidean
import time
from collections import defaultdict
import itertools
import random


class RobustPuzzleSolver:
    def __init__(
        self,
        puzzle_data_file="puzzle_complete_data.json",
        stats_file="puzzle_stats_summary.json",
        grid_width=6,
        grid_height=4,
        max_time=90,
    ):
        """
        Initialise le solveur de puzzle robuste

        Args:
            puzzle_data_file: Fichier JSON contenant les données complètes des pièces
            stats_file: Fichier JSON contenant les statistiques pour détecter les anomalies
            grid_width: Largeur de la grille (nombre de colonnes)
            grid_height: Hauteur de la grille (nombre de lignes)
            max_time: Temps maximum en secondes (90s = 1min30)
        """
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.total_pieces = grid_width * grid_height
        self.max_time = max_time
        self.start_time = None

        # Poids pour le matching
        self.COLOR_WEIGHT = 0.8  # 80% pour les couleurs
        self.SHAPE_WEIGHT = 0.2  # 20% pour la forme

        # Charger les données
        with open(puzzle_data_file, "r") as f:
            self.pieces_data = json.load(f)

        with open(stats_file, "r") as f:
            self.stats = json.load(f)

        print(f"Chargé {len(self.pieces_data)} pièces pour une grille {grid_width}x{grid_height}")

        # Grille de solution
        self.solution_grid = [[None for _ in range(grid_width)] for _ in range(grid_height)]
        self.pieces_used = set()
        self.piece_rotations = {}

        # Pièces suspectes et scores de fiabilité
        self.suspicious_pieces = set()
        self.piece_reliability_scores = {}

        # Catégorie de chaque pièce : "corner", "edge" ou "inner"
        self.piece_category = {}

        # Détecter les anomalies d'abord (pour remplir piece_reliability_scores)
        self.detect_anomalies()
        # Puis analyser les pièces (coins, bords, intérieures) et remplir piece_category
        self.analyze_pieces()

    def detect_anomalies(self):
        """Détecte les pièces potentiellement mal détectées"""
        print("\n=== DÉTECTION D'ANOMALIES ===")

        # Calculer les moyennes et écarts-types des aires
        all_areas = defaultdict(list)
        for piece in self.pieces_data:
            for side_name, side_data in piece["sides"].items():
                if "aire_sous_courbe" in side_data:
                    all_areas[side_name].append(abs(side_data["aire_sous_courbe"]))

        # Statistiques pour chaque côté
        side_stats = {}
        for side_name, areas in all_areas.items():
            side_stats[side_name] = {
                "mean": np.mean(areas),
                "std": np.std(areas),
                "median": np.median(areas),
            }

        # Détecter les anomalies
        anomaly_threshold = 2.5  # Nombre d'écarts-types pour considérer comme anomalie

        for piece in self.pieces_data:
            piece_score = 100  # Score de fiabilité initial
            anomalies = []

            # Vérifier les côtés incomplets
            expected_sides = ["haut", "droite", "bas", "gauche"]
            for side in expected_sides:
                if side not in piece["sides"]:
                    anomalies.append(f"Côté {side} manquant")
                    piece_score -= 20

            # Vérifier les aires anormales et autres anomalies
            for side_name, side_data in piece["sides"].items():
                # Aire anormale
                if "aire_sous_courbe" in side_data:
                    area = abs(side_data["aire_sous_courbe"])
                    stats = side_stats[side_name]
                    z_score = abs(area - stats["mean"]) / (stats["std"] + 1e-6)
                    if z_score > anomaly_threshold:
                        anomalies.append(
                            f"Aire {side_name} anormale: {area:.1f} (z-score: {z_score:.1f})"
                        )
                        piece_score -= 10

                # Cohérence des genres
                if "gender" in side_data and self.count_neutral_sides(piece) == 4:
                    anomalies.append("4 côtés neutres détectés")
                    piece_score -= 30

                # Segments de couleur manquants
                if "segments" not in side_data or not side_data["segments"]:
                    anomalies.append(f"Segments de couleur manquants pour {side_name}")
                    piece_score -= 15

            self.piece_reliability_scores[piece["filename"]] = piece_score

            if anomalies:
                self.suspicious_pieces.add(piece["filename"])
                print(f"\n{piece['filename']} - Score: {piece_score}/100")
                for anomaly in anomalies:
                    print(f"  - {anomaly}")

        print(f"\nPièces suspectes détectées: {len(self.suspicious_pieces)}/{len(self.pieces_data)}")

    def count_neutral_sides(self, piece):
        """Compte le nombre de côtés neutres d'une pièce"""
        count = 0
        for side_data in piece["sides"].values():
            if side_data.get("gender") == "neutre":
                count += 1
        return count

    def analyze_pieces(self):
        """Analyse les pièces pour identifier les coins, bords et pièces intérieures,
        et remplit self.piece_category[filename]"""
        self.corner_pieces = []
        self.edge_pieces = []
        self.inner_pieces = []

        for piece in self.pieces_data:
            neutral_sides = []
            for side_name, side_data in piece["sides"].items():
                if side_data.get("gender") == "neutre":
                    neutral_sides.append(side_name)

            filename = piece["filename"]
            reliability = self.piece_reliability_scores.get(filename, 100)
            piece_info = {
                "filename": filename,
                "neutral_sides": neutral_sides,
                "piece_data": piece,
                "reliability": reliability,
            }

            # Catégoriser
            if len(neutral_sides) == 2 and self.are_sides_adjacent(neutral_sides):
                self.corner_pieces.append(piece_info)
                self.piece_category[filename] = "corner"
            elif len(neutral_sides) == 1:
                self.edge_pieces.append(piece_info)
                self.piece_category[filename] = "edge"
            else:
                self.inner_pieces.append(piece_info)
                self.piece_category[filename] = "inner"

        # Trier par fiabilité (les plus fiables d'abord)
        self.corner_pieces.sort(key=lambda x: x["reliability"], reverse=True)
        self.edge_pieces.sort(key=lambda x: x["reliability"], reverse=True)
        self.inner_pieces.sort(key=lambda x: x["reliability"], reverse=True)

        print(f"\nAnalyse des pièces:")
        print(f"  - Coins: {len(self.corner_pieces)}")
        print(f"  - Bords: {len(self.edge_pieces)}")
        print(f"  - Intérieures: {len(self.inner_pieces)}")

    def are_sides_adjacent(self, sides):
        """Vérifie si deux côtés sont adjacents"""
        adjacent_pairs = [
            ["haut", "droite"],
            ["droite", "bas"],
            ["bas", "gauche"],
            ["gauche", "haut"],
        ]
        return sorted(sides) in [sorted(pair) for pair in adjacent_pairs]

    def get_side_after_rotation(self, side, rotation):
        """Obtient le nouveau côté après rotation (0, 90, 180, 270 degrés)"""
        sides = ["haut", "droite", "bas", "gauche"]
        idx = sides.index(side)
        return sides[(idx + rotation // 90) % 4]

    def get_opposite_side(self, side):
        """Obtient le côté opposé"""
        opposites = {"haut": "bas", "bas": "haut", "gauche": "droite", "droite": "gauche"}
        return opposites[side]

    def calculate_side_compatibility(self, piece1, side1, piece2, side2, rotation2=0):
        """
        Calcule la compatibilité entre deux côtés de pièces

        Returns:
            float: Score de compatibilité (plus bas = meilleur)
        """
        side2_rotated = self.get_side_after_rotation(side2, rotation2)

        if isinstance(piece1, dict) and "piece_data" in piece1:
            piece1_data = piece1["piece_data"]
        else:
            piece1_data = piece1

        if isinstance(piece2, dict) and "piece_data" in piece2:
            piece2_data = piece2["piece_data"]
        else:
            piece2_data = piece2

        side1_data = piece1_data["sides"].get(side1)
        side2_data = piece2_data["sides"].get(side2_rotated)

        if not side1_data or not side2_data:
            return float("inf")

        # Vérification des genres
        gender1 = side1_data.get("gender")
        gender2 = side2_data.get("gender")
        if gender1 == "neutre" and gender2 == "neutre":
            pass
        elif (gender1 == "male" and gender2 == "femelle") or (
            gender1 == "femelle" and gender2 == "male"
        ):
            pass
        else:
            return float("inf")

        # Comparer les couleurs
        color_score = self.compare_side_colors(side1_data, side2_data)

        # Comparer les formes
        area1 = side1_data.get("aire_sous_courbe", 0)
        area2 = side2_data.get("aire_sous_courbe", 0)
        shape_score = abs(area1 + area2)

        normalized_color_score = color_score / 50.0
        avg_area = (abs(area1) + abs(area2)) / 2
        normalized_shape_score = shape_score / (avg_area + 1e-6)

        total_score = (
            self.COLOR_WEIGHT * normalized_color_score
            + self.SHAPE_WEIGHT * normalized_shape_score
        )

        is_suspicious = (
            piece1_data["filename"] in self.suspicious_pieces
            or piece2_data["filename"] in self.suspicious_pieces
        )
        if is_suspicious:
            total_score *= 0.7

        return total_score

    def compare_side_colors(self, side1_data, side2_data):
        """
        Compare les couleurs de deux côtés.
        """
        segments1 = side1_data.get("segments", [])
        segments2 = side2_data.get("segments", [])
        if not segments1 or not segments2:
            return 100

        segments2_reversed = list(reversed(segments2))
        num_segments = min(len(segments1), len(segments2_reversed))
        if num_segments == 0:
            return 100

        total_diff = 0
        valid_comparisons = 0
        for i in range(num_segments):
            color1 = segments1[i].get("color_rgb", [128, 128, 128])
            color2 = segments2_reversed[i].get("color_rgb", [128, 128, 128])
            color_diff = euclidean(color1, color2)
            if sum(color1) > 30 and sum(color2) > 30:
                total_diff += color_diff
                valid_comparisons += 1

        return (total_diff / valid_comparisons) if valid_comparisons > 0 else 100

    def check_timeout(self):
        """Vérifie si le temps maximum est dépassé"""
        if self.start_time is None:
            return False
        return (time.time() - self.start_time) > self.max_time

    def get_position_constraints(self, row, col):
        """Obtient les contraintes pour une position donnée"""
        constraints = {}

        # Contraintes de bord (genre neutre sur le contour)
        if row == 0:
            constraints["haut"] = "neutre"
        if row == self.grid_height - 1:
            constraints["bas"] = "neutre"
        if col == 0:
            constraints["gauche"] = "neutre"
        if col == self.grid_width - 1:
            constraints["droite"] = "neutre"

        # Contraintes des pièces adjacentes
        if row > 0 and self.solution_grid[row - 1][col]:
            constraints["haut_piece"] = self.solution_grid[row - 1][col]
        if row < self.grid_height - 1 and self.solution_grid[row + 1][col]:
            constraints["bas_piece"] = self.solution_grid[row + 1][col]
        if col > 0 and self.solution_grid[row][col - 1]:
            constraints["gauche_piece"] = self.solution_grid[row][col - 1]
        if col < self.grid_width - 1 and self.solution_grid[row][col + 1]:
            constraints["droite_piece"] = self.solution_grid[row][col + 1]

        return constraints

    def evaluate_piece_at_position(self, piece, row, col, rotation, constraints):
        """Évalue une pièce à une position avec une rotation donnée"""
        score = 0

        if isinstance(piece, dict) and "piece_data" in piece:
            piece_data = piece["piece_data"]
        else:
            piece_data = piece

        # Pénalité légère pour pièces suspectes
        if piece_data["filename"] in self.suspicious_pieces:
            score += 10

        # Vérification des contraintes de bord
        for side, required_gender in constraints.items():
            if side in ["haut", "bas", "gauche", "droite"]:
                rotated_side = self.get_side_after_rotation(side, rotation)
                side_data = piece_data["sides"].get(rotated_side, {})
                if side_data.get("gender") != required_gender:
                    return float("inf")

        # Compatibilité avec les pièces adjacentes
        num_adjacent = 0
        for direction, adjacent_piece_info in constraints.items():
            if direction.endswith("_piece"):
                num_adjacent += 1
                side = direction.replace("_piece", "")
                adjacent_piece = adjacent_piece_info["piece_data"]
                adjacent_rotation = self.piece_rotations.get(adjacent_piece["filename"], 0)
                opposite_side = self.get_opposite_side(side)
                compatibility = self.calculate_side_compatibility(
                    piece_data, side, adjacent_piece, opposite_side, adjacent_rotation
                )
                if compatibility == float("inf"):
                    return float("inf")
                score += compatibility

        if num_adjacent > 0:
            score = score / num_adjacent

        return score

    def get_corner_rotation_for_position(self, corner_piece, row, col):
        """Détermine la rotation nécessaire pour un coin à une position donnée"""
        neutral_sides = corner_piece["neutral_sides"]
        if row == 0 and col == 0:
            required_sides = ["haut", "gauche"]
        elif row == 0 and col == self.grid_width - 1:
            required_sides = ["haut", "droite"]
        elif row == self.grid_height - 1 and col == 0:
            required_sides = ["bas", "gauche"]
        else:  # dernier coin
            required_sides = ["bas", "droite"]

        for rotation in [0, 90, 180, 270]:
            rotated_sides = [self.get_side_after_rotation(side, rotation) for side in neutral_sides]
            if set(rotated_sides) == set(required_sides):
                return rotation
        return 0

    def place_piece(self, piece_data, row, col, rotation):
        """Place une pièce à une position donnée"""
        self.solution_grid[row][col] = {"piece_data": piece_data, "rotation": rotation}
        self.pieces_used.add(piece_data["filename"])
        self.piece_rotations[piece_data["filename"]] = rotation

    def remove_piece(self, row, col):
        """Retire une pièce d'une position"""
        cell = self.solution_grid[row][col]
        if cell:
            piece_data = cell["piece_data"]
            self.pieces_used.remove(piece_data["filename"])
            del self.piece_rotations[piece_data["filename"]]
            self.solution_grid[row][col] = None

    def solve_puzzle(self):
        """Résout le puzzle avec gestion du temps et des anomalies"""
        print("\n=== RÉSOLUTION DU PUZZLE ===")
        print(f"Pondération: {self.COLOR_WEIGHT*100}% couleur, {self.SHAPE_WEIGHT*100}% forme")
        self.start_time = time.time()

        # Essayer avec différents coins de départ (jusqu'à 3 pièces de coin)
        corner_attempts = min(len(self.corner_pieces), 3)

        for corner_idx in range(corner_attempts):
            if self.check_timeout():
                break

            # Réinitialiser la grille
            self.solution_grid = [[None for _ in range(self.grid_width)] for _ in range(self.grid_height)]
            self.pieces_used.clear()
            self.piece_rotations.clear()

            # Placer le coin choisi en (0,0)
            first_corner = self.corner_pieces[corner_idx]
            rotation = self.get_corner_rotation_for_position(first_corner, 0, 0)
            self.place_piece(first_corner["piece_data"], 0, 0, rotation)

            print(f"\nEssai {corner_idx + 1} avec le coin: {first_corner['filename']} (fiabilité: {first_corner['reliability']}%)")

            # Backtracking avec catégories
            if self.solve_recursive_with_timeout():
                print("\n✓ Puzzle résolu avec succès (backtracking)!")
                # Remplir les positions vides qui subsistent
                self.fill_empty_positions()
                return True

        # Mode agressif si backtracking échoue
        print("\n⚠ Passage en mode agressif...")
        self.solve_aggressive()
        # Au terme de l’agressif, remplir explicitement tout le reste
        self.fill_empty_positions()
        return False

    def solve_recursive_with_timeout(self):
        """Résolution récursive en respectant timeout et contraintes de catégorie"""
        if self.check_timeout():
            return False

        # Trouver la prochaine position vide
        for row in range(self.grid_height):
            for col in range(self.grid_width):
                if self.solution_grid[row][col] is None:
                    # Déterminer la catégorie requise pour cette position
                    if (row in [0, self.grid_height - 1]) and (col in [0, self.grid_width - 1]):
                        required_category = "corner"
                    elif row in [0, self.grid_height - 1] or col in [0, self.grid_width - 1]:
                        required_category = "edge"
                    else:
                        required_category = "inner"

                    # Filtrer les pièces disponibles par catégorie
                    available_pieces = [
                        p
                        for p in self.pieces_data
                        if p["filename"] not in self.pieces_used
                        and self.piece_category.get(p["filename"], "inner") == required_category
                    ]

                    # Si aucune pièce de la bonne catégorie, fallback à toutes les restants
                    if not available_pieces:
                        available_pieces = [
                            p for p in self.pieces_data if p["filename"] not in self.pieces_used
                        ]

                    # Trier par fiabilité (plus fiable d'abord)
                    available_pieces.sort(
                        key=lambda p: self.piece_reliability_scores.get(p["filename"], 100),
                        reverse=True,
                    )

                    # Générer et trier candidats (pièce, rotation, score)
                    candidates = []
                    for piece in available_pieces:
                        for rotation in [0, 90, 180, 270]:
                            constraints = self.get_position_constraints(row, col)
                            piece_wrapper = {"piece_data": piece}
                            score = self.evaluate_piece_at_position(
                                piece_wrapper, row, col, rotation, constraints
                            )
                            if score < float("inf"):
                                candidates.append((piece, rotation, score))

                    # Trier par score (plus bas = meilleur)
                    candidates.sort(key=lambda x: x[2])

                    # Ajuster nombre de candidats selon le temps écoulé
                    elapsed = time.time() - self.start_time
                    if elapsed > self.max_time * 0.8:
                        max_cand = 2
                    elif elapsed > self.max_time * 0.5:
                        max_cand = 3
                    else:
                        max_cand = 5

                    # Tester les meilleurs candidats
                    for i, (piece, rotation, score) in enumerate(candidates[:max_cand]):
                        if self.check_timeout():
                            return False

                        # Parfois, pour diversifier, prendre un candidat « un peu moins bon »
                        if i > 2 and len(self.pieces_used) < self.total_pieces * 0.3:
                            if len(candidates) > 10:
                                idx = random.randint(5, min(10, len(candidates) - 1))
                                piece, rotation, score = candidates[idx]

                        self.place_piece(piece, row, col, rotation)
                        if self.solve_recursive_with_timeout():
                            return True
                        self.remove_piece(row, col)

                    return False

        # Si toutes les positions sont remplies selon backtracking, succès
        return True

    def solve_aggressive(self):
        """Mode agressif: place toutes les pièces restantes, en respectant la meilleure compatibilité possible"""
        print("Placement agressif des pièces restantes...")

        available_pieces = [p for p in self.pieces_data if p["filename"] not in self.pieces_used]
        available_pieces.sort(
            key=lambda p: self.piece_reliability_scores.get(p["filename"], 100),
            reverse=True,
        )

        for row in range(self.grid_height):
            for col in range(self.grid_width):
                if self.solution_grid[row][col] is None and available_pieces:
                    best_piece = None
                    best_rot = 0
                    best_score = float("inf")
                    for piece in available_pieces:
                        for rotation in [0, 90, 180, 270]:
                            score = self.evaluate_piece_at_position(
                                {"piece_data": piece},
                                row,
                                col,
                                rotation,
                                self.get_position_constraints(row, col),
                            )
                            if score < best_score:
                                best_score = score
                                best_piece = piece
                                best_rot = rotation

                    if best_piece:
                        self.place_piece(best_piece, row, col, best_rot)
                        available_pieces.remove(best_piece)

        return True

    def fill_empty_positions(self):
        """
        Remplit explicitement toutes les positions vides restantes
        en utilisant les pièces qui n'ont pas encore été placées,
        en respectant la catégorie au maximum (corner->corners, edge->edges).
        """
        # D'abord lister les positions vides par catégorie
        empty_corners = []
        empty_edges = []
        empty_inner = []
        for r in range(self.grid_height):
            for c in range(self.grid_width):
                if self.solution_grid[r][c] is None:
                    if (r in [0, self.grid_height - 1]) and (c in [0, self.grid_width - 1]):
                        empty_corners.append((r, c))
                    elif (r in [0, self.grid_height - 1]) or (c in [0, self.grid_width - 1]):
                        empty_edges.append((r, c))
                    else:
                        empty_inner.append((r, c))

        # Pièces restantes
        remaining = [p for p in self.pieces_data if p["filename"] not in self.pieces_used]

        # Fonction utilitaire pour placer sur une liste de positions à partir d'une catégorie donnée
        def place_for_positions(positions, category):
            nonlocal remaining
            for (r, c) in positions:
                # Filtrer pièces restantes par catégorie
                candidates = [
                    p for p in remaining if self.piece_category.get(p["filename"], "") == category
                ]
                # Si aucun candidat, prendre tous
                if not candidates:
                    candidates = remaining[:]
                # Choisir le meilleur candidat pour cette case
                best_piece = None
                best_rot = 0
                best_score = float("inf")
                for piece in candidates:
                    for rotation in [0, 90, 180, 270]:
                        score = self.evaluate_piece_at_position(
                            {"piece_data": piece},
                            r,
                            c,
                            rotation,
                            self.get_position_constraints(r, c),
                        )
                        if score < best_score:
                            best_score = score
                            best_piece = piece
                            best_rot = rotation
                if best_piece:
                    self.place_piece(best_piece, r, c, best_rot)
                    remaining = [p for p in remaining if p["filename"] != best_piece["filename"]]

        # Placer corners vides avec pièces de catégorie "corner"
        place_for_positions(empty_corners, "corner")
        # Placer edges vides avec pièces de catégorie "edge"
        place_for_positions(empty_edges, "edge")
        # Enfin, remplir le reste (inner ou pièces restantes)
        place_for_positions(empty_inner, "inner")

    def get_solution_summary(self):
        """Retourne un résumé de la solution"""
        solution = {
            "grid": [],
            "pieces_info": [],
            "empty_positions": [],
            "completion_time": time.time() - self.start_time if self.start_time else 0,
            "weights": {"color": self.COLOR_WEIGHT, "shape": self.SHAPE_WEIGHT},
        }

        for row in range(self.grid_height):
            grid_row = []
            for col in range(self.grid_width):
                cell = self.solution_grid[row][col]
                if cell:
                    piece_name = cell["piece_data"]["filename"]
                    rotation = cell["rotation"]
                    grid_row.append(f"{piece_name}:{rotation}°")
                    solution["pieces_info"].append(
                        {
                            "filename": piece_name,
                            "position": [row, col],
                            "rotation": rotation,
                            "is_suspicious": piece_name in self.suspicious_pieces,
                            "reliability_score": self.piece_reliability_scores.get(
                                piece_name, 100
                            ),
                        }
                    )
                else:
                    grid_row.append("VIDE")
                    solution["empty_positions"].append([row, col])
            solution["grid"].append(grid_row)

        return solution

    def print_solution_grid(self):
        """Affiche la grille de solution de manière lisible"""
        print("\n=== GRILLE DE SOLUTION ===")
        for row in range(self.grid_height):
            row_str = "|"
            for col in range(self.grid_width):
                cell = self.solution_grid[row][col]
                if cell:
                    piece_name = cell["piece_data"]["filename"].replace("piece_", "P")
                    rotation = cell["rotation"]
                    if cell["piece_data"]["filename"] in self.suspicious_pieces:
                        row_str += f" {piece_name}:{rotation:3d}°* |"
                    else:
                        row_str += f" {piece_name}:{rotation:3d}° |"
                else:
                    row_str += "  VIDE   |"
            print(row_str)
            if row < self.grid_height - 1:
                print("-" * len(row_str))
        print("\n* = Pièce suspecte")

    def save_solution(self, output_file="puzzle_solution_robust.json"):
        """Sauvegarde la solution dans un fichier JSON"""
        solution = self.get_solution_summary()
        solution["suspicious_pieces"] = list(self.suspicious_pieces)
        solution["reliability_scores"] = self.piece_reliability_scores

        with open(output_file, "w") as f:
            json.dump(solution, f, indent=2)

        print(f"\nSolution sauvegardée dans {output_file}")
        print(f"Temps de résolution: {solution['completion_time']:.2f}s")
        print(f"Positions vides: {len(solution['empty_positions'])}")


# === Utilisation ===
if __name__ == "__main__":
    solver = RobustPuzzleSolver(
        puzzle_data_file="puzzle_complete_data.json",
        stats_file="puzzle_stats_summary.json",
        grid_width=6,
        grid_height=4,
        max_time=90,  # 1min30
    )

    solver.solve_puzzle()  # backtracking puis agressif, puis fill_empty_positions intégré

    solver.print_solution_grid()
    solver.save_solution("puzzle_solution_robust.json")

    solution = solver.get_solution_summary()
    print(f"\nNombre de pièces placées: {len(solution['pieces_info'])}/{solver.total_pieces}")
    print(f"Temps total: {solution['completion_time']:.2f}s")

    if solution["empty_positions"]:
        print(f"\nPositions vides: {solution['empty_positions']}")

    if solver.suspicious_pieces:
        print(f"\nPièces suspectes placées:")
        for piece_info in solution["pieces_info"]:
            if piece_info["is_suspicious"]:
                print(
                    f"  - {piece_info['filename']} à la position {piece_info['position']} "
                    f"(score: {piece_info['reliability_score']}%)"
                )
