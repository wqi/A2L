import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import vizdoom as vzd

from src.common.custom_game import CustomGame

'''
This script is used to extract a birds-eye map view around the spawn point within a specified map.

Configuration is performed via command line arguments specified below.
'''


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Paths
    parser.add_argument('--wad-dir', type=str, default='./data/maps/test',
                        help='Path to dir containing maps')
    parser.add_argument('--wad-id', type=int, required=True,
                        help='WAD ID of map to generate automap for')
    parser.add_argument('--out-dir', type=str, default='.',
                        help='Path to dir where output automap should be saved')

    args = parser.parse_args()
    return args


def setup_game(wad_dir, wad_id):
    # Set up VizDoom Game
    game = CustomGame()
    game.load_config('../data/configs/human.cfg')
    game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
    game.set_render_weapon(False)
    game.set_window_visible(False)
    game.set_automap_mode(vzd.AutomapMode.WHOLE)
    game.set_automap_rotate(True)
    game.add_game_args("+am_followplayer 1")
    game.add_game_args("+viz_am_scale 0.8")
    game.add_game_args("+am_backcolor 888888")
    game.add_game_args("+freelook 1")
    game.set_mode(vzd.Mode.SPECTATOR)

    # Load generated map from WAD
    wad_path = os.path.join(wad_dir, '{}.wad'.format(wad_id)) # NOQA
    game.set_doom_scenario_path(wad_path)
    game.init()
    game.send_game_command("iddqd")
    return game


def save_automap(wad_dir, wad_id, out_dir):
    # Initialize game
    game = setup_game(wad_dir, wad_id)

    # Get automap
    state = game.get_state()
    automap = np.rollaxis(state.automap_buffer, 0, 3)
    plt.imshow(automap)

    # Save automap
    out_path = os.path.join(out_dir, 'automap_{}.png'.format(wad_id))
    plt.imsave(out_path, automap)


if __name__ == "__main__":
    args = parse_arguments()
    save_automap(args.wad_dir, args.wad_id, args.out_dir)
