import json
import os


class PuzzleJSONMerger:
    def __init__(self,
                 classification_file="puzzle_classification.json",
                 normalized_file="puzzle_all_sides_features.json",
                 colors_file="puzzle_colors.json"):
        """
        Initialise le fusionneur avec les chemins des 3 fichiers JSON

        Args:
            classification_file: JSON avec les types de côtés (neutre/mâle/femelle)
            normalized_file: JSON avec les points normalisés et aires sous courbe
            colors_file: JSON avec les couleurs des segments
        """
        self.classification_file = classification_file
        self.normalized_file = normalized_file
        self.colors_file = colors_file

        # Charger les données
        self.classification_data = self.load_json(classification_file)
        self.normalized_data = self.load_json(normalized_file)
        self.colors_data = self.load_json(colors_file)

        print(f"Données chargées:")
        print(f"  - Classification: {len(self.classification_data)} pièces")
        print(f"  - Normalisation: {len(self.normalized_data)} pièces")
        print(f"  - Couleurs: {len(self.colors_data)} pièces")

    def load_json(self, filename):
        """Charge un fichier JSON"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            print(f"✓ Chargé: {filename}")
            return data
        except FileNotFoundError:
            print(f"✗ Erreur: Fichier {filename} non trouvé")
            return []
        except json.JSONDecodeError as e:
            print(f"✗ Erreur JSON dans {filename}: {e}")
            return []

    def find_piece_by_filename(self, data_list, filename):
        """Trouve une pièce dans une liste de données par son nom de fichier"""
        for item in data_list:
            if item.get('filename') == filename:
                return item
        return None

    def merge_all_data(self):
        """
        Fusionne les 3 sources de données en un seul JSON consolidé

        Returns:
            list: Liste des pièces avec toutes leurs données fusionnées
        """
        merged_data = []

        # Utiliser la liste de classification comme base (car elle contient tous les côtés)
        for classification_piece in self.classification_data:
            filename = classification_piece['filename']

            # Trouver les données correspondantes dans les autres sources
            normalized_piece = self.find_piece_by_filename(self.normalized_data, filename)
            colors_piece = self.find_piece_by_filename(self.colors_data, filename)

            # Créer l'entrée fusionnée
            merged_piece = {
                'filename': filename,
                'image_dimensions': classification_piece.get('image_dimensions'),
                'piece_center': classification_piece.get('piece_center'),
                'sides': {}
            }

            side_names = ["haut", "droite", "bas", "gauche"]

            for side_name in side_names:
                side_data = {}

                # Données de classification (seulement le gender)
                if side_name in classification_piece.get('sides', {}):
                    classification_side = classification_piece['sides'][side_name]
                    side_data.update({
                        'gender': classification_side.get('gender')
                    })

                # Données de normalisation (tout garder)
                if normalized_piece and side_name in normalized_piece.get('sides', {}):
                    normalized_side = normalized_piece['sides'][side_name]
                    side_data.update({
                        'aire_sous_courbe': normalized_side.get('aire_sous_courbe'),
                        'points_normalises': normalized_side.get('points_normalises'),
                        'longueur': normalized_side.get('longueur'),
                        'angle_rotation': normalized_side.get('angle_rotation')
                    })

                # Données de couleurs (tout garder)
                if colors_piece and side_name in colors_piece.get('sides', {}):
                    colors_side = colors_piece['sides'][side_name]
                    side_data.update({
                        'segments': colors_side.get('segments', []),
                        'total_segments': colors_side.get('total_segments', 0)
                    })

                # Ajouter le côté seulement s'il a des données
                if side_data:
                    merged_piece['sides'][side_name] = side_data

            merged_data.append(merged_piece)

        return merged_data

    def save_merged_data(self, output_file="puzzle_complete_data.json"):
        """
        Sauvegarde les données fusionnées dans un nouveau fichier JSON

        Args:
            output_file: Nom du fichier de sortie
        """
        merged_data = self.merge_all_data()

        with open(output_file, 'w') as f:
            json.dump(merged_data, f, indent=2)

        print(f"\n✓ Données fusionnées sauvegardées dans: {output_file}")
        print(f"  Nombre de pièces: {len(merged_data)}")

        # Statistiques
        total_sides = 0
        sides_with_colors = 0
        sides_with_normalization = 0
        sides_with_classification = 0

        for piece in merged_data:
            for side_name, side_data in piece['sides'].items():
                total_sides += 1
                if 'segments' in side_data:
                    sides_with_colors += 1
                if 'points_normalises' in side_data:
                    sides_with_normalization += 1
                if 'gender' in side_data:
                    sides_with_classification += 1

        print(f"  Statistiques:")
        print(f"    - Total côtés: {total_sides}")
        print(f"    - Avec classification: {sides_with_classification}")
        print(f"    - Avec normalisation: {sides_with_normalization}")
        print(f"    - Avec couleurs: {sides_with_colors}")

        return merged_data

    def validate_merged_data(self, merged_data):
        """
        Valide la cohérence des données fusionnées

        Args:
            merged_data: Données fusionnées à valider
        """
        print(f"\n=== VALIDATION DES DONNÉES ===")

        issues = []
        complete_pieces = 0

        for piece in merged_data:
            filename = piece['filename']
            piece_issues = []
            piece_complete = True

            for side_name in ["haut", "droite", "bas", "gauche"]:
                if side_name not in piece['sides']:
                    piece_issues.append(f"Côté {side_name} manquant")
                    piece_complete = False
                else:
                    side_data = piece['sides'][side_name]

                    # Vérifier les données essentielles
                    if 'gender' not in side_data:
                        piece_issues.append(f"Côté {side_name}: classification manquante")
                        piece_complete = False

                    if 'points_normalises' not in side_data:
                        piece_issues.append(f"Côté {side_name}: points normalisés manquants")
                        piece_complete = False

                    if 'segments' not in side_data:
                        piece_issues.append(f"Côté {side_name}: couleurs manquantes")
                        piece_complete = False

            if piece_complete:
                complete_pieces += 1

            if piece_issues:
                issues.append(f"{filename}: {', '.join(piece_issues)}")

        print(f"Pièces complètes: {complete_pieces}/{len(merged_data)}")

        if issues:
            print(f"\nProblèmes détectés:")
            for issue in issues[:10]:  # Afficher max 10 problèmes
                print(f"  - {issue}")
            if len(issues) > 10:
                print(f"  ... et {len(issues) - 10} autres problèmes")
        else:
            print("✓ Aucun problème détecté")

    def get_piece_summary(self, filename):
        """
        Affiche un résumé des données d'une pièce spécifique

        Args:
            filename: Nom du fichier de la pièce
        """
        merged_data = self.merge_all_data()

        piece = self.find_piece_by_filename(merged_data, filename)
        if not piece:
            print(f"Pièce {filename} non trouvée")
            return

        print(f"\n=== RÉSUMÉ: {filename} ===")
        print(f"Dimensions: {piece.get('image_dimensions')}")
        print(f"Centre: {piece.get('piece_center')}")

        for side_name, side_data in piece['sides'].items():
            print(f"\nCôté {side_name}:")
            print(f"  - Genre: {side_data.get('gender', 'N/A')}")
            print(f"  - Longueur: {side_data.get('longueur', 'N/A'):.1f}px")
            print(f"  - Aire sous courbe: {side_data.get('aire_sous_courbe', 'N/A')}")
            print(f"  - Segments couleur: {side_data.get('total_segments', 'N/A')}")

            if side_data.get('segments'):
                # Afficher quelques couleurs
                segments = side_data['segments'][:3]
                colors_preview = [seg.get('color_hex', 'N/A') for seg in segments]
                print(f"  - Couleurs (3 premiers): {', '.join(colors_preview)}")

    def export_summary_stats(self, output_file="puzzle_stats_summary.json"):
        """
        Exporte des statistiques résumées sur toutes les pièces
        """
        merged_data = self.merge_all_data()

        stats = {
            'total_pieces': len(merged_data),
            'gender_distribution': {'neutre': 0, 'male': 0, 'femelle': 0},
            'average_areas': {'haut': 0, 'droite': 0, 'bas': 0, 'gauche': 0},
            'pieces_with_complete_data': 0
        }

        area_sums = {'haut': [], 'droite': [], 'bas': [], 'gauche': []}

        for piece in merged_data:
            complete_data = True

            for side_name in ["haut", "droite", "bas", "gauche"]:
                if side_name in piece['sides']:
                    side_data = piece['sides'][side_name]

                    # Compter les genres
                    gender = side_data.get('gender')
                    if gender in stats['gender_distribution']:
                        stats['gender_distribution'][gender] += 1

                    # Aires sous courbe
                    area = side_data.get('aire_sous_courbe')
                    if area is not None:
                        area_sums[side_name].append(area)

                    # Vérifier données complètes
                    if not all(key in side_data for key in ['gender', 'points_normalises', 'segments']):
                        complete_data = False
                else:
                    complete_data = False

            if complete_data:
                stats['pieces_with_complete_data'] += 1

        # Calculer les moyennes des aires
        for side_name in area_sums:
            if area_sums[side_name]:
                stats['average_areas'][side_name] = sum(area_sums[side_name]) / len(area_sums[side_name])

        with open(output_file, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"\n✓ Statistiques sauvegardées dans: {output_file}")
        return stats


# === Utilisation ===
if __name__ == "__main__":
    # Créer le fusionneur
    merger = PuzzleJSONMerger(
        classification_file="puzzle_classification.json",
        normalized_file="puzzle_all_sides_features.json",
        colors_file="puzzle_colors.json"
    )

    # Fusionner et sauvegarder
    merged_data = merger.save_merged_data("puzzle_complete_data.json")

    # Valider les données
    merger.validate_merged_data(merged_data)

    # Exporter des statistiques
    stats = merger.export_summary_stats("puzzle_stats_summary.json")

    # Afficher un résumé d'une pièce (exemple)
    if merged_data:
        first_piece = merged_data[0]['filename']
        merger.get_piece_summary(first_piece)

    print(f"\n=== FICHIERS GÉNÉRÉS ===")
    print(f"✓ puzzle_complete_data.json - Données complètes fusionnées")
    print(f"✓ puzzle_stats_summary.json - Statistiques résumées")
    print(f"\nLe fichier principal 'puzzle_complete_data.json' contient toutes vos données consolidées!")