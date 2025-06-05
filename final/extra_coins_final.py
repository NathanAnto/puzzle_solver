import cv2
import numpy as np
import os
import math
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from itertools import combinations, permutations


# --- Line and Corner Detection Logic (Helper Functions) ---

def find_aligned_segments(contour_points, sampling_step=8, min_points=5, angle_threshold=0.2, distance_threshold=10):
    """
    Find aligned segments in contour points.
    Note: cv2.approxPolyDP is generally recommended over this custom approach for robustness.
    """
    if len(contour_points) < min_points * sampling_step:
        return []

    lines = []
    current_segment = []

    # Sample points using the sampling_step parameter
    sampled_points = [contour_points[i][0] for i in range(0, len(contour_points), sampling_step)]

    if len(sampled_points) < 2:
        return []

    for i, point in enumerate(sampled_points):
        if len(current_segment) < 2:
            current_segment.append(point)
            continue

        if is_point_aligned_with_segment(current_segment, point, angle_threshold, distance_threshold):
            current_segment.append(point)
        else:
            if len(current_segment) >= min_points:
                lines.append((current_segment[0], current_segment[-1]))
            current_segment = [current_segment[-1], point]

    if len(current_segment) >= min_points:
        lines.append((current_segment[0], current_segment[-1]))

    return lines


def is_point_aligned_with_segment(segment_points, new_point, angle_threshold, distance_threshold):
    if len(segment_points) < 2:
        return True

    start_point = segment_points[0]
    end_point = segment_points[-1]

    segment_vector = end_point - start_point
    segment_length = np.linalg.norm(segment_vector)

    if segment_length < 1e-6:
        return np.linalg.norm(new_point - start_point) < distance_threshold

    segment_direction = segment_vector / segment_length
    new_vector = new_point - start_point

    distance = np.abs(np.cross(segment_direction, new_point - start_point))

    if distance > distance_threshold:
        return False

    vec_to_new_point = new_point - end_point
    len_to_new_point = np.linalg.norm(vec_to_new_point)

    if len_to_new_point < 1e-6:
        return True

    dir_to_new_point = vec_to_new_point / len_to_new_point

    dot_product = np.dot(segment_direction, dir_to_new_point)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    angle_diff = np.arccos(dot_product)

    return angle_diff <= angle_threshold


def find_line_intersection(line1, line2):
    p1, p2 = line1
    p3, p4 = line2
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    x3, y3 = float(p3[0]), float(p3[1])
    x4, y4 = float(p4[0]), float(p4[1])

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom

    ix = x1 + t * (x2 - x1)
    iy = y1 + t * (y2 - y1)

    return np.array([ix, iy])


def calculate_angle_between_lines(line1, line2):
    dir1 = np.array(line1[1]) - np.array(line1[0])
    dir2 = np.array(line2[1]) - np.array(line2[0])
    norm1 = np.linalg.norm(dir1)
    norm2 = np.linalg.norm(dir2)

    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0

    dir1 = dir1 / norm1
    dir2 = dir2 / norm2

    dot_product = np.dot(dir1, dir2)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    angle_rad = np.arccos(abs(dot_product))
    return np.degrees(angle_rad)


def is_point_near_line_segments(point, line1, line2, max_distance=20):
    def point_to_line_segment_distance(p, a, b):
        if np.array_equal(a, b):
            return np.linalg.norm(p - a)
        line_vec = b - a
        point_vec = p - a
        line_len_sq = np.dot(line_vec, line_vec)
        t = np.dot(point_vec, line_vec) / line_len_sq
        t = max(0, min(1, t))
        closest = a + t * line_vec
        return np.linalg.norm(p - closest)

    dist1 = point_to_line_segment_distance(point, np.array(line1[0]), np.array(line1[1]))
    dist2 = point_to_line_segment_distance(point, np.array(line2[0]), np.array(line2[1]))

    return dist1 <= max_distance and dist2 <= max_distance


def find_corner_intersections(lines, angle_tolerance=15, corner_max_distance=30):
    corners = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            line1 = lines[i]
            line2 = lines[j]
            angle = calculate_angle_between_lines(line1, line2)

            if abs(angle - 90) <= angle_tolerance:
                intersection = find_line_intersection(line1, line2)
                if intersection is not None:
                    if is_point_near_line_segments(intersection, line1, line2, corner_max_distance):
                        corners.append({
                            'point': intersection, 'lines_indices': (i, j),
                            'angle': angle, 'line1_pts': line1, 'line2_pts': line2
                        })
    return corners


def point_to_line_distance(point, line_start, line_end):
    """Calculate the perpendicular distance from a point to a line segment."""
    if np.array_equal(line_start, line_end):
        return np.linalg.norm(point - line_start)

    line_vec = line_end - line_start
    point_vec = point - line_start
    line_len_sq = np.dot(line_vec, line_vec)

    if line_len_sq < 1e-10:
        return np.linalg.norm(point - line_start)

    t = np.dot(point_vec, line_vec) / line_len_sq
    t = max(0, min(1, t))  # Clamp to line segment
    closest = line_start + t * line_vec
    return np.linalg.norm(point - closest)


def count_points_near_quadrilateral(quad_vertices, all_corner_points, max_distance=25):
    """
    Count how many corner points are near the edges of a quadrilateral.
    """
    if len(quad_vertices) != 4:
        return 0

    count = 0

    # Check each corner point against all 4 edges of the quadrilateral
    for corner_info in all_corner_points:
        corner_point = corner_info['point']
        is_near_any_edge = False

        # Check distance to each edge of the quadrilateral
        for i in range(4):
            edge_start = quad_vertices[i]
            edge_end = quad_vertices[(i + 1) % 4]  # Next vertex, wrapping around

            distance = point_to_line_distance(corner_point, edge_start, edge_end)

            if distance <= max_distance:
                is_near_any_edge = True
                break

        if is_near_any_edge:
            count += 1

    return count


def is_valid_quadrilateral(vertices):
    """
    Check if 4 points form a valid quadrilateral (not degenerate, reasonable shape).
    """
    if len(vertices) != 4:
        return False

    # Check that no three points are collinear
    for i in range(4):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % 4]
        p3 = vertices[(i + 2) % 4]

        # Calculate cross product to check collinearity
        v1 = p2 - p1
        v2 = p3 - p1
        cross = np.cross(v1, v2)

        if abs(cross) < 1e-6:  # Nearly collinear
            return False

    # Check that the quadrilateral has reasonable proportions
    # Calculate area using shoelace formula
    area = 0
    for i in range(4):
        j = (i + 1) % 4
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]
    area = abs(area) / 2

    if area < 100:  # Too small
        return False

    return True


def find_best_quadrilateral_from_corners(corners, max_distance_to_edge=25):
    """
    Find the quadrilateral formed by 4 corners that maximizes the number of
    corner points lying near its edges.
    """
    if len(corners) < 4:
        print(f"  Not enough corners for quadrilateral selection: {len(corners)}")
        return corners  # Return what we have

    corner_points = [corner['point'] for corner in corners]

    best_score = -1
    best_quad_indices = None

    print(f"  Evaluating {len(list(combinations(range(len(corners)), 4)))} possible quadrilaterals...")

    # Try all combinations of 4 corners
    for quad_indices in permutations(range(len(corners)), 4):
        quad_vertices = [corner_points[i] for i in quad_indices]

        # Check if this forms a valid quadrilateral
        if not is_valid_quadrilateral(quad_vertices):
            continue

        # Count how many corner points are near this quadrilateral's edges
        score = count_points_near_quadrilateral(quad_vertices, corners, max_distance_to_edge)

        # Add a small bonus for more rectangular shapes (optional)
        angle_bonus = 0
        for i in range(4):
            p1 = quad_vertices[i]
            p2 = quad_vertices[(i + 1) % 4]
            p3 = quad_vertices[(i + 2) % 4]

            v1 = p1 - p2
            v2 = p3 - p2

            if np.linalg.norm(v1) > 1e-6 and np.linalg.norm(v2) > 1e-6:
                v1_norm = v1 / np.linalg.norm(v1)
                v2_norm = v2 / np.linalg.norm(v2)

                dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
                angle = np.degrees(np.arccos(abs(dot_product)))

                # Bonus for angles close to 90 degrees
                angle_bonus += max(0, 1 - abs(angle - 90) / 90) * 0.1

        total_score = score + angle_bonus

        if total_score > best_score:
            best_score = total_score
            best_quad_indices = quad_indices
            print(
                f"    New best quadrilateral: indices {quad_indices}, score {total_score:.2f} ({score} points + {angle_bonus:.2f} angle bonus)")

    if best_quad_indices is None:
        print("  No valid quadrilateral found, using first 4 corners")
        return corners[:4]

    print(f"  Selected quadrilateral with {len(best_quad_indices)} corners, score: {best_score:.2f}")

    # Return the 4 corners that form the best quadrilateral
    selected_corners = [corners[i] for i in best_quad_indices]

    # Sort corners in a consistent order (e.g., clockwise from top-left)
    # Calculate centroid
    centroid = np.mean([corner['point'] for corner in selected_corners], axis=0)

    # Sort by angle from centroid
    def angle_from_centroid(corner):
        point = corner['point']
        return np.arctan2(point[1] - centroid[1], point[0] - centroid[0])

    selected_corners.sort(key=angle_from_centroid)

    return selected_corners


def export_corners_to_json(all_piece_corners, output_folder=None, filename="puzzle_corners.json"):
    """
    Export corner coordinates to a JSON file.

    Args:
        all_piece_corners: Dictionary with piece names as keys and corner coordinates as values
        output_folder: Folder to save the JSON file (optional)
        filename: Name of the JSON file
    """
    # Convert the data to a more structured format
    export_data = {}

    for piece_name, corners in all_piece_corners.items():
        # Remove file extension and create piece identifier
        piece_id = os.path.splitext(piece_name)[0]

        # Ensure we have exactly 4 corners
        if len(corners) >= 4:
            corner_data = []
            for i, (x, y) in enumerate(corners[:4]):
                corner_data.append({
                    "corner_id": i,
                    "x": float(x),
                    "y": float(y)
                })

            export_data[piece_id] = {
                "original_filename": piece_name,
                "num_corners": len(corner_data),
                "corners": corner_data
            }
        else:
            print(f"Warning: Piece {piece_name} has only {len(corners)} corners, expected 4")
            corner_data = []
            for i, (x, y) in enumerate(corners):
                corner_data.append({
                    "corner_id": i,
                    "x": float(x),
                    "y": float(y)
                })

            export_data[piece_id] = {
                "original_filename": piece_name,
                "num_corners": len(corner_data),
                "corners": corner_data
            }

    # Determine output path
    if output_folder:
        json_path = os.path.join(output_folder, filename)
    else:
        json_path = filename

    # Write to JSON file
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        print(f"\nCorners exported successfully to: {json_path}")
        print(f"Exported data for {len(export_data)} pieces")

        # Print a sample of the exported data
        if export_data:
            sample_piece = next(iter(export_data.keys()))
            print(f"\nSample data for piece '{sample_piece}':")
            print(json.dumps(export_data[sample_piece], indent=2))

    except Exception as e:
        print(f"Error writing JSON file: {e}")

    return json_path


def plot_piece_analysis(ax, image_binary, lines, corners, selected_corners, filename):
    """
    Plot the analysis of a single puzzle piece on a matplotlib axis.
    """
    # Display the binary image
    ax.imshow(image_binary, cmap='gray')
    ax.set_title(f'{filename}', fontsize=10)
    ax.axis('off')

    # Color palette for lines
    line_colors = ['cyan', 'magenta', 'yellow', 'purple', 'orange', 'lime', 'red', 'blue']

    # Draw detected lines
    for i, (start, end) in enumerate(lines):
        color = line_colors[i % len(line_colors)]
        ax.plot([start[0], end[0]], [start[1], end[1]],
                color=color, linewidth=2, alpha=0.8)

    # Draw the selected quadrilateral
    if len(selected_corners) >= 4:
        quad_points = [corner['point'] for corner in selected_corners[:4]]
        quad_points.append(quad_points[0])  # Close the polygon
        quad_x = [p[0] for p in quad_points]
        quad_y = [p[1] for p in quad_points]
        ax.plot(quad_x, quad_y, color='magenta', linewidth=3, alpha=0.9)

    # Draw all corners (small red circles)
    for corner_info in corners:
        point = corner_info['point']
        ax.plot(point[0], point[1], 'ro', markersize=4, alpha=0.6)

    # Draw selected corners (larger green circles with numbers)
    for i, corner_info in enumerate(selected_corners):
        point = corner_info['point']
        # Green circle with white border
        ax.plot(point[0], point[1], 'go', markersize=8, markeredgecolor='white', markeredgewidth=2)
        # Add number label
        ax.text(point[0] + 10, point[1] - 10, str(i),
                color='red', fontsize=8, fontweight='bold')


# --- Main Processing Function ---
def process_puzzle_pieces(input_folder, output_folder=None,
                          # Parameters for contour approximation
                          epsilon_factor=0.015,
                          min_line_length=20,
                          # Parameters for corner detection
                          corner_angle_tolerance=25,
                          corner_max_distance=40,
                          force_four_corners=True,
                          quad_edge_distance_threshold=25):
    """
    Process puzzle piece images to detect line segments and corner intersections.
    Uses matplotlib for visualization instead of OpenCV.
    """
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)

    all_piece_corners = {}
    piece_data = []  # Store data for plotting

    # Get all valid image files
    valid_files = [f for f in os.listdir(input_folder)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not valid_files:
        print("No valid image files found!")
        return all_piece_corners

    for filename in valid_files:
        image_path = os.path.join(input_folder, filename)
        image_binary = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image_binary is None:
            print(f"Error reading: {filename}")
            continue

        print(f"\nProcessing: {filename}")

        contours, _ = cv2.findContours(image_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if not contours:
            print(f"No contours found in {filename}")
            continue

        largest_contour = max(contours, key=cv2.contourArea)

        # --- Line Segment Detection using Douglas-Peucker (approxPolyDP) ---
        perimeter = cv2.arcLength(largest_contour, True)
        epsilon = epsilon_factor * perimeter
        approx_poly_vertices = cv2.approxPolyDP(largest_contour, epsilon, True)

        # Extract line segments from the approximated polygon
        lines = []
        num_vertices = len(approx_poly_vertices)
        for i in range(num_vertices):
            p1 = approx_poly_vertices[i][0]
            p2 = approx_poly_vertices[(i + 1) % num_vertices][0]

            if np.linalg.norm(p1 - p2) >= min_line_length:
                lines.append((p1, p2))

        print(f"  Found {len(approx_poly_vertices)} vertices in approximated polygon.")
        print(f"  Extracted {len(lines)} line segments (min length {min_line_length}).")

        # --- Find Corner Intersections from these lines ---
        corners = find_corner_intersections(
            lines,
            angle_tolerance=corner_angle_tolerance,
            corner_max_distance=corner_max_distance
        )

        print(f"  Initial corner detection found: {len(corners)} corners")

        all_corners = corners.copy()
        # --- Find Best Quadrilateral from Corners ---
        if force_four_corners and len(corners) >= 4:
            selected_corners = find_best_quadrilateral_from_corners(
                corners,
                max_distance_to_edge=quad_edge_distance_threshold
            )
        elif len(corners) > 4:
            selected_corners = corners[:4]
        else:
            selected_corners = corners

        print(f"  Final selected corners: {len(selected_corners)}")

        # Store corner points
        detected_corner_points = [tuple(corner['point'].astype(int)) for corner in selected_corners]
        all_piece_corners[filename] = detected_corner_points

        # Store data for plotting
        piece_data.append({
            'filename': filename,
            'image': image_binary,
            'lines': lines,
            'all_corners': all_corners,
            'selected_corners': selected_corners
        })

    # Create matplotlib figure with subplots
    n_pieces = len(piece_data)
    if n_pieces == 0:
        print("No pieces to display!")
        return all_piece_corners

    # Calculate subplot grid (try to make it roughly square)
    cols = int(np.ceil(np.sqrt(n_pieces)))
    rows = int(np.ceil(n_pieces / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    fig.suptitle('Puzzle Pieces Corner Detection', fontsize=16)

    # Handle case where we have only one subplot
    if n_pieces == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()

    # Plot each piece
    for i, data in enumerate(piece_data):
        ax = axes[i] if isinstance(axes, (list, np.ndarray)) else axes
        plot_piece_analysis(
            ax,
            data['image'],
            data['lines'],
            data['all_corners'],
            data['selected_corners'],
            data['filename']
        )

    # Hide empty subplots
    for i in range(n_pieces, len(axes) if isinstance(axes, (list, np.ndarray)) else 1):
        if isinstance(axes, (list, np.ndarray)) and i < len(axes):
            axes[i].axis('off')

    plt.tight_layout()

    # Save the figure if output folder is specified
    if output_folder:
        output_path = os.path.join(output_folder, 'all_pieces_analysis.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved complete analysis to {output_path}")

    plt.show()

    # Export corners to JSON
    if all_piece_corners:
        json_path = export_corners_to_json(all_piece_corners, output_folder, "puzzle_corners.json")

    return all_piece_corners


# Example usage:
if __name__ == "__main__":
    input_folder = "piece_noir_blanc"
    output_folder = "corners_output"

    EPSILON_FACTOR = 0.0075
    MIN_LINE_LEN = 22
    CORNER_ANGLE_TOL = 25
    CORNER_MAX_DIST = 5
    QUAD_EDGE_DISTANCE_THRESHOLD = 5  # How close corners must be to quad edges

    detected_corners_data = process_puzzle_pieces(
        input_folder,
        output_folder,
        epsilon_factor=EPSILON_FACTOR,
        min_line_length=MIN_LINE_LEN,
        corner_angle_tolerance=CORNER_ANGLE_TOL,
        corner_max_distance=CORNER_MAX_DIST,
        force_four_corners=True,
        quad_edge_distance_threshold=QUAD_EDGE_DISTANCE_THRESHOLD
    )

    print("\n" + "=" * 50)
    print("PROCESSING COMPLETE")
    print("=" * 50)
    print(f"Total pieces processed: {len(detected_corners_data)}")
    for piece_name, corners in detected_corners_data.items():
        print(f"  {piece_name}: {len(corners)} corners")
    print("=" * 50)