import cv2
import math
import numpy as np
import random
import vizdoom as vzd

from os.path import join


def navigate_beeline(simple_map):
    actions = [0, 0, 0]

    # Spin around if no beacon is present in map
    no_beacon = np.sum(simple_map == 4) == 0
    if no_beacon:
        actions[1] = 1.0
        return actions

    # Take action to move towards goal if present
    center = simple_map.shape[0] / 2.0
    beacon_locs = np.where(simple_map == 4)
    min_dist = 9999

    for l in zip(beacon_locs[0], beacon_locs[1]):
        xdiff = l[0] - center
        ydiff = l[1] - center
        d = abs(xdiff) + abs(ydiff)
        if (d < min_dist):
            min_dist = d
            actions[2] = 1.0
            if (abs(ydiff) > 3):
                if (ydiff < 0):
                    actions[0] = 1.0
                else:
                    actions[1] = 1.0

    return actions


def update_map(state, height=240, width=320,
               map_size=256, map_scale=3, fov=90.0,
               beacon_scale=50, pick_new_goal=False,
               only_visible_beacons=True, curr_goal=None,
               explored_goals={}, nodes={}, edges={}):
    # Extract agent state from game
    depth_buffer = state.depth_buffer
    player_x = state.game_variables[0]
    player_y = state.game_variables[1]
    player_angle = state.game_variables[2]

    # Initialize maps
    canvas_size = 2*map_size + 1
    simple_map = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

    # Compute upper left and upper right extreme coordinates
    r = canvas_size
    offset = 225

    y1 = int(r * math.cos(math.radians(offset + player_angle)))
    x1 = int(r * math.sin(math.radians(offset + player_angle)))

    y2 = int(r * math.cos(math.radians(offset + player_angle - fov)))
    x2 = int(r * math.sin(math.radians(offset + player_angle - fov)))

    # Draw FOV boundaries
    _, p1, p2 = cv2.clipLine((0, 0, canvas_size, canvas_size),
                             (map_size, map_size),
                             (map_size + x1, map_size + y1))
    _, p3, p4 = cv2.clipLine((0, 0, canvas_size, canvas_size),
                             (map_size, map_size),
                             (map_size + x2, map_size + y2))

    # Ray cast from eye line to project depth map into 2D ray points
    game_unit = 100.0/14
    ray_cast = (depth_buffer[height//2] * game_unit)/float(map_scale)

    ray_points = [(map_size, map_size)]
    for i in range(10, canvas_size-10):
        d = ray_cast[int(float(width)/canvas_size * i - 1)]
        theta = (float(i)/canvas_size * fov)

        ray_y = int(d * math.sin(math.radians(offset - theta))) + map_size
        ray_x = int(d * math.cos(math.radians(offset - theta))) + map_size

        _, _, p = cv2.clipLine((0, 0, canvas_size, canvas_size),
                               (map_size, map_size),
                               (ray_y, ray_x))
        ray_points.append(p)

    # Fill free space on 2D map with colour
    cv2.fillPoly(simple_map, np.array([ray_points], dtype=np.int32), (1, 1, 1))

    quantized_x = int(player_x/beacon_scale) * beacon_scale
    quantized_y = int(player_y/beacon_scale) * beacon_scale
    beacon_radius = 10

    # Get beacons within range of current agent position
    beacons = []
    for bnx in range(-beacon_radius, beacon_radius+1):
        for bny in range(-beacon_radius, beacon_radius+1):
            beacon_x = quantized_x + bnx * beacon_scale
            beacon_y = quantized_y + bny * beacon_scale
            beacons.append((beacon_x, beacon_y))

    # Compute set of visible beacons and draw onto the map
    visible_beacons_world = []
    for b in beacons:
        # Insert nodes and edges for beacons
        if b not in nodes:
            nodes[b] = True

        neighbors = [(b[0], b[1] - beacon_scale),
                     (b[0], b[1] + beacon_scale),
                     (b[0] - beacon_scale, b[1]),
                     (b[0] + beacon_scale, b[1]),
                     (b[0] - beacon_scale, b[1] - beacon_scale),
                     (b[0] + beacon_scale, b[1] + beacon_scale),
                     (b[0] - beacon_scale, b[1] + beacon_scale),
                     (b[0] + beacon_scale, b[1] - beacon_scale)]

        for n in neighbors:
            if n in visible_beacons_world:
                if (b, n) not in edges:
                    edges[(b, n)] = True
                    edges[(n, b)] = True

        # Draw beacons into map
        object_relative_x = -b[0] + player_x
        object_relative_y = -b[1] + player_y

        rotated_x = math.cos(math.radians(-player_angle)) * object_relative_x - math.sin(math.radians(-player_angle)) * object_relative_y # NOQA
        rotated_y = math.sin(math.radians(-player_angle)) * object_relative_x + math.cos(math.radians(-player_angle)) * object_relative_y # NOQA

        rotated_x = int(rotated_x/map_scale + map_size)
        rotated_y = int(rotated_y/map_scale + map_size)

        if (rotated_x >= 0 and rotated_x < canvas_size and
           rotated_y >= 0 and rotated_y < canvas_size):
            object_id = 3
            show = True
            if simple_map[rotated_x, rotated_y] == 0:
                show = (only_visible_beacons is not True)
            else:
                visible_beacons_world.append((b[0], b[1]))

            if show:
                simple_map[rotated_x, rotated_y] = object_id

    # Pick new goal from unexplored visible beacons if required
    if pick_new_goal:
        unexplored_beacons = []
        for b in visible_beacons_world:
            if b not in explored_goals:
                unexplored_beacons.append(b)

        if len(unexplored_beacons) > 0:
            beacon_idx = random.randint(0, len(unexplored_beacons)-1)
            curr_goal = unexplored_beacons[beacon_idx]
            explored_goals[curr_goal] = True
        else:
            curr_goal = None

    # Draw current goal location on map
    if curr_goal is not None:
        object_relative_x = -curr_goal[0] + player_x
        object_relative_y = -curr_goal[1] + player_y

        rotated_x = math.cos(math.radians(-player_angle)) * object_relative_x - math.sin(math.radians(-player_angle)) * object_relative_y # NOQA
        rotated_y = math.sin(math.radians(-player_angle)) * object_relative_x + math.cos(math.radians(-player_angle)) * object_relative_y # NOQA

        rotated_x = int(rotated_x/map_scale + map_size)
        rotated_y = int(rotated_y/map_scale + map_size)

        if (rotated_x >= 0 and rotated_x < canvas_size and
           rotated_y >= 0 and rotated_y < canvas_size):
            object_id = 4
            if simple_map[rotated_x, rotated_y] > 0:
                simple_map[rotated_x, rotated_y] = object_id

    return simple_map, curr_goal


def explore_map_random_policy(config, wad_dir, wad_id, nodes={}, edges={}, explored_goals={}):
    wad_path = join(wad_dir, '{}.wad'.format(wad_id))

    # Set up Doom enviroinment
    game = vzd.DoomGame()
    game.load_config(config)
    game.set_doom_scenario_path(wad_path)
    game.set_window_visible(False)
    game.init()
    game.send_game_command("iddqd")
    game.new_episode()

    # Play until the game (episode) is over
    step = 0
    curr_goal = None
    explored_goals = {}

    while not game.is_episode_finished():
        # Pick new local goal in visibility every 50 steps
        pick_new_goal = True if step % 50 == 0 else False

        # Update map from current POV
        state = game.get_state()
        simple_map, curr_goal = update_map(state, pick_new_goal=pick_new_goal,
                                           curr_goal=curr_goal, explored_goals=explored_goals,
                                           nodes=nodes, edges=edges)

        # Take action towards current goal
        action = navigate_beeline(simple_map)
        game.make_action(action)
        step = step + 1
    game.close()

    return nodes, edges, explored_goals
