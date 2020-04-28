import cv2
import math
import numpy as np
import os
import pickle
import vizdoom as vzd

from os import listdir
from os.path import isfile, join
from vizdoom import ScreenResolution

from common.custom_game import CustomGame

'''
Collection of utility functions that are employed throughout the A2L repo.
'''


def get_sorted_wad_ids(wad_dir):
    '''
    Get sorted list of WAD ids contained within specified directory.
    '''
    all_files = [f for f in listdir(wad_dir) if isfile(join(wad_dir, f))]
    wad_files = [f for f in all_files if f.endswith('wad')]
    wad_ids = [int(f.split('.wad')[0]) for f in wad_files]
    wad_ids.sort()

    return wad_ids


def setup_game(wad_dir, wad_id, visible=False, localization_noise=0, pose_noise=0):
    '''
    Set up custom VizDoom game with specified configuration.
    '''
    # Set up VizDoom Game
    game = CustomGame(localization_noise, pose_noise)
    game.load_config('../data/configs/delta.cfg')
    game.set_screen_resolution(ScreenResolution.RES_320X240)
    game.set_render_weapon(False)
    game.set_window_visible(visible)

    game.set_button_max_value(vzd.Button.TURN_LEFT_RIGHT_DELTA, 10)

    # Load generated map from WAD
    wad_path = os.path.join(wad_dir, '{}.wad'.format(wad_id)) # NOQA
    game.set_doom_scenario_path(wad_path)
    game.init()
    return game


def compute_map_state(state, height=240, width=320, map_size=65, map_scale=8,
                      fov=90.0):
    '''
    Get local geometric map from current game state.
    '''
    # Extract agent state from game
    depth_buffer = state.depth_buffer

    # Initialize maps
    map_size = map_size - 1
    canvas_size = 2*map_size + 1
    vis_map = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    simple_map = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

    # Ray cast from eye line to project depth map into 2D ray points
    offset = 225
    game_unit = 100.0/14
    ray_cast = (depth_buffer[height//2] * game_unit)/float(map_scale)
    ray_points = [(map_size, map_size)]
    for i in range(10, canvas_size-10):
        d = ray_cast[int(round(float(width)/canvas_size * i - 1))]
        theta = (float(i)/canvas_size * fov)

        ray_y = int(round(d * math.sin(math.radians(offset - theta)))) + map_size
        ray_x = int(round(d * math.cos(math.radians(offset - theta)))) + map_size

        _, _, p = cv2.clipLine((0, 0, canvas_size, canvas_size),
                               (map_size, map_size), (ray_y, ray_x))
        ray_points.append(p)

    # Label known free space on 2D map with value 2
    cv2.fillPoly(vis_map, np.array([ray_points], dtype=np.int32), (255, 255, 255)) # NOQA
    cv2.fillPoly(simple_map, np.array([ray_points], dtype=np.int32), 2)

    # Label known obstacles on 2D map with value 1
    for point in ray_points[1:]:
        if point[1] > 0:
            cond1 = point[0] + 1 < canvas_size
            cond2 = point[1] + 1 < canvas_size
            cond3 = point[0] - 1 > 0
            cond4 = point[1] - 1 > 0

            simple_map[point[1], point[0]] = 1
            if cond1:
                simple_map[point[1], point[0] + 1] = 1
            if cond2:
                simple_map[point[1] + 1, point[0]] = 1
            if cond3:
                simple_map[point[1], point[0] - 1] = 1
            if cond4:
                simple_map[point[1] - 1, point[0]] = 1
            if cond1 and cond2:
                simple_map[point[1] + 1, point[0] + 1] = 1
            if cond3 and cond4:
                simple_map[point[1] - 1, point[0] - 1] = 1
            if cond2 and cond3:
                simple_map[point[1] + 1, point[0] - 1] = 1
            if cond1 and cond4:
                simple_map[point[1] - 1, point[0] + 1] = 1

    # Crop generated maps
    vis_map = vis_map[:map_size + 1, map_size - 32:map_size + 32 + 1, :]
    simple_map = simple_map[:map_size + 1, map_size - 32:map_size + 32 + 1]
    # plt.imshow(simple_map)
    # plt.show()
    return vis_map, simple_map


def compute_map(game, height=240, width=320, map_size=65, map_scale=8,
                fov=90.0):
    '''
    (Convenience Function) Get local geometric map from current game.
    '''
    state = game.get_state()
    vis_map, simple_map = compute_map_state(state, height, width, map_size, map_scale, fov)
    return vis_map, simple_map


def get_valid_locations(beacon_dir, wad_id):
    '''
    Get list of valid spawn locations (in open space) from directory of saved beacons.
    '''
    node_path = os.path.join(beacon_dir, '{}.pkl'.format(wad_id))
    with open(node_path, 'rb') as handle:
        nodes = pickle.load(handle)
    return list(nodes.keys())


def abs_to_map_pos(game, abs_pos, map_size=64, map_scale=8):
    '''
    Convert absolute coordinate to local map coordinate.
    '''
    player_x, player_y, player_angle = game.get_agent_location()
    rel_x = -abs_pos[0] + player_x
    rel_y = -abs_pos[1] + player_y

    rotated_x = math.cos(math.radians(-player_angle)) * rel_x - math.sin(math.radians(-player_angle)) * rel_y # NOQA
    rotated_y = math.sin(math.radians(-player_angle)) * rel_x + math.cos(math.radians(-player_angle)) * rel_y # NOQA
    rotated_x = int(round(rotated_x/map_scale + map_size))
    rotated_y = int(round(rotated_y/map_scale + map_size)) - 32
    return (rotated_x, rotated_y)


def abs_to_global_map_pos(game, abs_pos, origin, global_map_size=1024, map_scale=8):
    '''
    Convert absolute coordinate to global map coordinate.
    '''
    global_x = int(round((origin[1] - abs_pos[1])/map_scale + global_map_size/2))
    global_y = int(round((abs_pos[0] - origin[0])/map_scale + global_map_size/2))
    return (global_x, global_y)


def global_map_to_abs_pos(game, map_pos, origin, global_map_size=1024, map_scale=8):
    '''
    Convert global map coordinate to absolute coordinate.
    '''
    abs_x = int(round((map_pos[1] - global_map_size/2) * map_scale + origin[0]))
    abs_y = int(round(((map_pos[0] - global_map_size/2) * map_scale - origin[1]) * -1))
    return(abs_x, abs_y)


def map_to_abs_pos(game, map_pos, map_size=64, map_scale=8):
    '''
    Convert local map coordinate to absolute coordinate.
    '''
    player_x, player_y, player_angle = game.get_agent_location()

    rotated_x = (map_pos[0] - map_size) * map_scale
    rotated_y = (map_pos[1] + 32 - map_size) * map_scale
    rel_x = math.cos(math.radians(player_angle)) * rotated_x - \
        math.sin(math.radians(player_angle)) * rotated_y
    rel_y = math.sin(math.radians(player_angle)) * rotated_x + \
        math.cos(math.radians(player_angle)) * rotated_y
    abs_pos_x = int(np.around(player_x - rel_x))
    abs_pos_y = int(np.around(player_y - rel_y))
    return (abs_pos_x, abs_pos_y)


def dist_to(game, point):
    '''
    Compute current distance from agent to specified point.
    '''
    player_x, player_y, _ = game.get_agent_location()

    diff = np.zeros(2)
    diff[0] = np.abs(point[0] - player_x)
    diff[1] = np.abs(point[1] - player_y)
    dist = np.linalg.norm(diff)

    return dist


def check_monster_collision(game):
    '''
    Check if agent is in collision state with monster.
    '''
    state = game.get_state()

    if state is None:
        return False

    player_x = game.get_game_variable(vzd.GameVariable.POSITION_X)
    player_y = game.get_game_variable(vzd.GameVariable.POSITION_Y)
    for l in state.labels:
            if l.object_name == 'Zombieman':
                mon_rel_x = l.object_position_x - player_x
                mon_rel_y = l.object_position_y - player_y
                if np.linalg.norm([mon_rel_x, mon_rel_y]) < 50:
                    return True
    return False


def get_angle_diff(player_angle, sample_angle):
    '''
    Compute smallest difference between two specified angles.
    '''
    angle_diff = sample_angle - player_angle
    if angle_diff < 0:
        counter_rotate = abs(angle_diff) > abs(angle_diff + 360)
        smallest_diff = angle_diff + 360 if counter_rotate else angle_diff
    else:
        counter_rotate = abs(angle_diff) > abs(angle_diff - 360)
        smallest_diff = angle_diff - 360 if counter_rotate else angle_diff
    return smallest_diff


def check_damage(game):
    '''
    Check if agent has taken damage in last timestep.
    Health is reset to max after every check to make sure agent doesn't die.
    '''
    took_damage = False
    health = game.get_game_variable(vzd.GameVariable.HEALTH)

    if health < 85:
        took_damage = True
    game.send_game_command("give health")
    return took_damage


def setup_trial(game, start_pos):
    '''
    Project segmented first-person view into top-down 2D space.
    '''
    game.new_episode()
    game.send_game_command('warp {} {}'.format(start_pos[0], start_pos[1]))

    # Idle for several actions to ensure warp completes
    for i in range(2):
        game.make_action([0, 0])
