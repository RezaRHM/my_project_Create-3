import numpy as np
import cv2
from heapq import heappop, heappush

# Returns a map of all walkable pixels in a maze, with "clearance_px" distance from obstacles
def compute_allowed_mask(maze, clearance_px):
    if clearance_px <= 0:
        return (maze == 255)

    free = (maze == 255).astype(np.uint8)                           # 1 = free, 0 = Obstacle
    dist = cv2.distanceTransform(free, cv2.DIST_L2, 5)     # distance map from nearest obstacle
    allowed = dist >= float(clearance_px)
    return allowed


def is_walkable(maze, position, allowed_mask=None):
    row, col = position
    if row < 0 or row >= maze.shape[0] or col < 0 or col >= maze.shape[1]:
        return False

    if allowed_mask is not None:
        return allowed_mask[row, col]

    return maze[row, col] == 255


def get_neighbors(maze, position, allowed_mask=None):
    row, col = position
    neighbors = []
    directions = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1)
    ]

    for dr, dc in directions:
        neighbor = (row + dr, col + dc)
        if is_walkable(maze, neighbor, allowed_mask):
            neighbors.append(neighbor)
    return neighbors


def movement_cost(current, neighbor, parent=None, turn_penalty=0.5):
    if abs(current[0] - neighbor[0]) + abs(current[1] - neighbor[1]) == 2:
        base_cost = np.sqrt(2)  # Diagonal movement
    else:
        base_cost = 1.0  # Straight movement

    if parent is None:
        return base_cost

    dir_in = (current[0] - parent[0], current[1] - parent[1])  # parent -> current
    dir_out = (neighbor[0] - current[0], neighbor[1] - current[1])  # current -> neighbor

    # penalize for change of direction
    if dir_in != dir_out:
        return base_cost + turn_penalty

    return base_cost


# heuristic function (Euclidean distance)
# This can be changed to other distance metrics if needed
def heuristic(pos, goal):
    return np.sqrt((pos[0] - goal[0])**2 + (pos[1] - goal[1])**2)


"""----------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------"""


def a_star(maze, start, goal, clearance_px=0, turn_penalty=0.5):
    allowed_mask = compute_allowed_mask(maze, clearance_px)

    if not allowed_mask[start] or not allowed_mask[goal]:
        print("Start or goal position is too close to obstacles!")
        return None

    open_set = []
    counter = 0
    heappush(open_set, (0, counter, start))

    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    closed_set = set()

    while open_set:
        current = heappop(open_set)[2]

        if current in closed_set:
            continue

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        closed_set.add(current)

        # Get parent for turn penalty calculation
        parent = came_from.get(current, None)

        for neighbor in get_neighbors(maze, current, allowed_mask):
            if neighbor in closed_set:
                continue

            # Calculate cost with turn penalty
            tentative_g = g_score[current] + movement_cost(current, neighbor, parent, turn_penalty)

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)

                counter += 1
                heappush(open_set, (f_score[neighbor], counter, neighbor))

    return None
