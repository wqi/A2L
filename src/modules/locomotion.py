import math
import random
import vizdoom as vzd

import common.util as util


def navigate_beeline(game, goal):
    '''
    Executes action to move agent towards goal, following beeline navigation policy.
    '''
    player_x, player_y, player_angle = game.get_agent_location()

    goal_angle_rad = math.atan2(goal[1] - player_y, goal[0] - player_x)
    goal_angle = math.degrees(goal_angle_rad)
    if goal_angle < 0:
        goal_angle = 360 + goal_angle

    if abs(goal_angle - player_angle) > 15:
        angle_diff = goal_angle - player_angle
        if angle_diff > 180:
            angle_diff -= 360
        elif angle_diff <= -180:
            angle_diff += 360
        action = [-angle_diff, 0]
    else:
        action = [0, 15]

    game.make_action(action)


def rotate_random_angle(game):
    '''
    Executes actions to rotate agent to random angle
    '''
    sample_angle = random.randint(0, 359)
    player_angle = game.get_game_variable(vzd.GameVariable.ANGLE)
    smallest_diff = util.get_angle_diff(player_angle, sample_angle)

    # Rotate agent to random angle
    while abs(smallest_diff) > 5:
        game.make_action([-smallest_diff, 0])
        player_angle = game.get_game_variable(vzd.GameVariable.ANGLE)
        smallest_diff = util.get_angle_diff(player_angle, sample_angle)
