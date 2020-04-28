import numpy as np

import common.pyastar as pyastar


def shortest_path_geometric(map, start, goal):
    '''
    Find shortest path to goal using geometric navigation cost map.
    '''
    # Check if start and goal locations are valid
    map[start[0], start[1]] = 2
    map[start[0], start[1] - 1] = 2
    map[start[0], start[1] + 1] = 2
    map[start[0] + 1, start[1]] = 2
    map[start[0] + 1, start[1] - 1] = 2
    map[start[0] + 1, start[1] + 1] = 2
    map[start[0] - 1, start[1]] = 2
    map[start[0] - 1, start[1] - 1] = 2
    map[start[0] - 1, start[1] + 1] = 2
    if map[goal[0], goal[1]] == 1:
        return []

    # Find shortest path using A star
    map = np.copy(map).astype(np.float32)
    map[np.where(map == 0)] = 3
    map[np.where(map == 1)] = 999
    map[np.where(map == 2)] = 1

    path = pyastar.astar_path(map, start, goal, allow_diagonal=True)
    return path


def shortest_path_affordance(map, start, goal):
    '''
    Find shortest path to goal using affordance-based navigation cost map.
    '''
    # Check if start and goal locations are valid
    map[start[0], start[1]] = 1
    map[start[0], start[1] - 1] = 1
    map[start[0], start[1] + 1] = 1
    map[start[0] + 1, start[1]] = 1
    map[start[0] + 1, start[1] - 1] = 1
    map[start[0] + 1, start[1] + 1] = 1
    map[start[0] - 1, start[1]] = 1
    map[start[0] - 1, start[1] - 1] = 1
    map[start[0] - 1, start[1] + 1] = 1

    # Find shortest path using A star
    map = np.copy(map).astype(np.float32)
    map = np.around(map)
    map[np.where(map == 0)] = 20  # Set unknown cells to higher cost

    path = pyastar.astar_path(map, start, goal, allow_diagonal=True)
    return path
