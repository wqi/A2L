import argparse
import os
import vizdoom as vzd

from src.common.custom_game import CustomGame # NOQA

'''
This script is used to explore VizDoom maps as an invincible human player.

Configuration is performed via command line arguments specified below.
'''


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Paths
    parser.add_argument('--wad-dir', type=str, default='./data/maps/test',
                        help='Path to dir containing maps')
    parser.add_argument('--cfg-path', type=str, default='./data/configs/default.cfg',
                        help='Path to VizDoom config file in player mode')
    parser.add_argument('--wad-id', type=int, required=True,
                        help='WAD ID of map to play')

    args = parser.parse_args()
    return args


def setup_player_game(wad_dir, cfg_path, wad_id):
    # Set up VizDoom Game
    game = CustomGame()
    game.load_config(cfg_path)
    game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
    game.set_render_weapon(False)
    game.set_window_visible(True)
    game.add_game_args("+freelook 1")
    game.set_mode(vzd.Mode.SPECTATOR)

    # Load generated map from WAD
    wad_path = os.path.join(wad_dir, '{}.wad'.format(wad_id)) # NOQA
    game.set_doom_scenario_path(wad_path)
    game.init()
    game.send_game_command("iddqd")
    return game


def play_map(wad_dir, cfg_path, wad_id):
    # Initialize game
    game = setup_player_game(wad_dir, cfg_path, wad_id)

    # Initialize variables
    total_steps = 0

    # Play game
    while not game.is_episode_finished():
        # Advance action
        game.advance_action()

        # Log current status
        if total_steps % 10 == 0:
            cur_pos = (game.get_game_variable(vzd.POSITION_X),
                       game.get_game_variable(vzd.POSITION_Y))
            print("Step #" + str(total_steps))
            print("Position:", cur_pos)
            print("=====================")

        total_steps += 1


if __name__ == "__main__":
    args = parse_arguments()
    play_map(args.wad_dir, args.cfg_path, args.wad_id)
