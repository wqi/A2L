import argparse
import cv2
import json
import math
import numpy as np
import os
import time
import vizdoom as vzd

from collections import defaultdict
from os.path import join

import common.util as util
from common.custom_game import CustomGame


'''
This script is used to evaluate human navigation performance. Participants are asked to navigate
to a relative goal shown in an ego-centric map. Experiments are specified and results are dumped
to disk in JSON format.

Configuration is performed via command line arguments specified below.
'''


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Paths
    parser.add_argument('--wad-dir', type=str, default='../data/maps/test',
                        help='Path to dir containing map files')
    parser.add_argument('--save-dir', type=str,
                        help='Path to dir where results and vids should be saved')
    parser.add_argument('--experiment-path', type=str,
                        default='../data/experiments/navigation/demo.json',
                        help='Path to file containing experimental setup')

    # Navigation Options
    parser.add_argument('--max-steps', type=int, default=1000,
                        help='Max number of steps for navigation trial')

    # Config Options
    parser.add_argument('--save-vid', action='store_true',
                        help='Specifies whether videos should be saved from every run')

    args = parser.parse_args()
    return args


def setup_player_game(wad_dir, wad_id):
    # Set up VizDoom Game
    game = CustomGame()
    game.load_config('../data/configs/default.cfg')
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_render_weapon(False)
    game.set_window_visible(True)
    game.add_game_args("+freelook 1")
    game.set_mode(vzd.Mode.SPECTATOR)

    # Load generated map from WAD
    wad_path = join(wad_dir, '{}.wad'.format(wad_id)) # NOQA
    game.set_doom_scenario_path(wad_path)
    game.init()
    return game


def build_goal_map(game, final_goal):
    # Build visualization for map of relative goal position
    goal_map = np.zeros((101, 101, 3), np.uint8)
    cv2.circle(goal_map, (50, 50), 1, (0, 255, 0), -1)

    player_x, player_y, player_angle = game.get_agent_location()
    rel_x = -final_goal[0] + player_x
    rel_y = -final_goal[1] + player_y

    rotated_x = math.cos(math.radians(-player_angle)) * rel_x - math.sin(math.radians(-player_angle)) * rel_y # NOQA
    rotated_y = math.sin(math.radians(-player_angle)) * rel_x + math.cos(math.radians(-player_angle)) * rel_y # NOQA
    rotated_x = int(round(rotated_x/25) + 50)
    rotated_y = int(round(rotated_y/25) + 50)
    rotated_x = min(100, max(0, rotated_x))
    rotated_y = min(100, max(0, rotated_y))
    cv2.circle(goal_map, (rotated_y, rotated_x), 2, (0, 0, 255), -1)

    return goal_map


def human_navigate(game, max_steps, final_goal, vid_path=None):
    # Set up video writer
    vid_out = None
    if vid_path is not None:
        vid_out = cv2.VideoWriter(vid_path,
                                  cv2.VideoWriter_fourcc(*'mp4v'),
                                  vzd.DEFAULT_TICRATE // 2, (640, 480))
    cv2.namedWindow('Map', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Map', 500, 500)

    # Initialize variables
    total_damage = 0
    total_steps = 0

    while not game.is_episode_finished() and total_steps < args.max_steps:
        # Get current state and buffer
        state = game.get_state()
        rgb_frame = np.rollaxis(state.screen_buffer, 0, 3)

        # Show goal relative to agent
        goal_map = build_goal_map(game, final_goal)
        cv2.imshow('Map', goal_map)
        cv2.waitKey(1)

        # Advance action
        game.advance_action()

        # Allow human time before episode starts
        if total_steps == 0:
            time.sleep(2)

        # Check for damage
        cur_damage = 100 - game.get_game_variable(vzd.HEALTH)
        if cur_damage > 15:
            total_damage += 20
            print('Total Damage: {}'.format(total_damage))
        if util.check_monster_collision(game):
            total_damage += 4
            print('Total Damage: {}'.format(total_damage))
        game.send_game_command("give health")

        # Check if goal is reached
        dist_to_goal = util.dist_to(game, final_goal)
        if dist_to_goal < 40:
            print('Goal reached')
            return {'success': 1, 'steps': total_steps, 'damage': total_damage}

        # Write to video if specified
        if vid_out:
            vis_buffer = np.zeros((480, 640, 3), dtype=np.uint8)
            rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
            vis_buffer[:480, :640, :] = rgb_frame
            vid_out.write(vis_buffer)

        total_steps += 1

    return {'success': 0, 'steps': total_steps, 'damage': total_damage}


def run_navigation_trials(args):
    # Create save directories for results and vids
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        if args.save_vid:
            vid_dir = join(args.save_dir, 'vids')
            os.makedirs(vid_dir, exist_ok=True)

    # Read saved experimental setup from disk
    with open(args.experiment_path, 'r') as fp:
        experiment_dict = json.load(fp)

    # Execute experiments for specified number of times
    result_dict = defaultdict(list)
    for wad_id, experiments in sorted(experiment_dict.items()):
        game = setup_player_game(args.wad_dir, wad_id)
        for idx, experiment in enumerate(experiments):
            vid_path = join(vid_dir, '{}_{}.mp4'.format(wad_id, idx)) if args.save_vid else None
            util.setup_trial(game, experiment['start'])
            result = human_navigate(game, args.max_steps, experiment['goal'], vid_path)
            result_dict[wad_id].append(result)
        game.close()

        # Save results from experiment
        if args.save_dir:
            result_path = join(args.save_dir, 'results.json')
            with open(result_path, 'w') as fp:
                json.dump(result_dict, fp, sort_keys=True, indent=4)


if __name__ == "__main__":
    args = parse_arguments()
    run_navigation_trials(args)
