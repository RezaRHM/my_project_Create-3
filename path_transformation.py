import numpy as np

# Detecting straight movement of 3 consecutive points
def straight_move(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    if y1 == y2 and y2 == y3:
        return True
    elif x1 == x2 and x1 == x3:
        return True

    return False

# Reduce the path to Straight and Diagonal lines
def path_reduction(path):
    reduced_path = [path[0]]
    i = 0

    try:
        while True:
                current_x, current_y = path[i]
                n_x, n_y = path[i + 2]

                if abs(n_x - current_x) == abs(n_y - current_y) or straight_move(path[i], path[i + 1], path[i + 2]):
                    i += 2
                else:
                    i += 1

                reduced_path.append(path[i])
    except IndexError:
        reduced_path.extend(path[i+1:])
        pass

    # Recursive function - repeats until further reduction is not possible
    if path != reduced_path:
        path = reduced_path.copy()
        return path_reduction(path)
    else:
        return reduced_path

# Calculating cm to pixel ratio
def px_to_cm(image, height_cm, width_cm):
    height_px, width_px = image.shape

    sh = height_cm / height_px
    sw = width_cm / width_px

    return sh, sw

# Returning an array of waypoints in polar coordinate (angle, magnitude)
def path_to_cm(path_px, maze=None, maze_shape_cm=None):
    sh, sw = np.round(px_to_cm(maze, maze_shape_cm[1], maze_shape_cm[0]), decimals = 2)
    magnitudes = []
    lines_mat = [(0, 1)]

    for i in range(len(path_px) - 1):                   # Calculating the absolute distance between each pair of consecutive action points
        dy_px = (path_px[i + 1][0] - path_px[i][0])
        dx_px = (path_px[i + 1][1] - path_px[i][1])

        movement_cm = (sh * dy_px, sw * dx_px)
        lines_mat.append(movement_cm)

        length_cm = np.sqrt(sum(element ** 2 for element in movement_cm))
        magnitudes.append(np.round(length_cm, decimals = 2))

    angles = []
    for i in range(len(lines_mat) - 1):            # Calculating the angle between each two consecutive pair of lines
        u, v = lines_mat[i], lines_mat[i + 1]
        angle_u = np.arctan2(u[0], u[1])
        angle_v = np.arctan2(v[0], v[1])

        angle = np.degrees(angle_v - angle_u)
        angle_diff = (angle + 180) % 360 - 180     # Normalizing angle to the range [-180, 180] for increasing simplicity

        angles.append(angle_diff)

    output = [(float(angles[i]), float(magnitudes[i])) for i in range(len(angles))]

    return output
