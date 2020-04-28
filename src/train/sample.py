import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import vizdoom as vzd

import common.util as util
import modules.affordance as affordance
import modules.locomotion as locomotion
import modules.planning as planning


def pick_goal(game, fs_map, explored_goals, beacon_radius=12,
              beacon_scale=40, map_scale=8, map_size=64):
    '''
    Pick random point in agent's FOV to set as sampling goal.
    '''
    # Recreate full map from partial free space map
    canvas_size = 2*map_size + 1
    full_map = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    full_map[:map_size + 1, map_size - 32:map_size + 32 + 1] = fs_map

    # Get current player position and angle
    player_x, player_y, player_angle = game.get_agent_location()
    quantized_x = int(round(player_x/beacon_scale)) * beacon_scale
    quantized_y = int(round(player_y/beacon_scale)) * beacon_scale

    # Get beacons within range of current agent position
    beacons = []
    for bnx in range(-beacon_radius, beacon_radius+1):
        for bny in range(-beacon_radius, beacon_radius+1):
            beacon_x = quantized_x + bnx * beacon_scale
            beacon_y = quantized_y + bny * beacon_scale
            beacons.append((beacon_x, beacon_y))

    # Compute set of visible beacons
    visible_beacons_world = []
    for b in beacons:
        object_relative_x = -b[0] + player_x
        object_relative_y = -b[1] + player_y
        rotated_x = math.cos(math.radians(-player_angle)) * object_relative_x - math.sin(math.radians(-player_angle)) * object_relative_y # NOQA
        rotated_y = math.sin(math.radians(-player_angle)) * object_relative_x + math.cos(math.radians(-player_angle)) * object_relative_y # NOQA
        rotated_x = int(round(rotated_x/map_scale + map_size))
        rotated_y = int(round(rotated_y/map_scale + map_size))

        if (rotated_x >= 0 and rotated_x < canvas_size and
           rotated_y >= 0 and rotated_y < canvas_size):
            if full_map[rotated_x, rotated_y] == 0:
                continue
            object_id = 3
            visible_beacons_world.append((b[0], b[1]))
            full_map[rotated_x, rotated_y] = object_id

    # Pick new goal from unexplored visible beacons if required
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

    # Compute relative goal
    rel_goal = None
    if curr_goal is not None:
        diff_x = curr_goal[0] - player_x
        diff_y = curr_goal[1] - player_y
        rel_goal = np.array([diff_x, diff_y])

    return curr_goal, rel_goal


def get_view_xy(state):
    '''
    Get absolute XY coordinates corresponding to each pixel in agent FOV
    '''
    # Set hyper-parameters
    width = 320.0
    fov = 110.0
    game_unit = 100.0/14

    # Get depth buffer and position from game state
    depth_buffer = state.depth_buffer
    player_x = state.game_variables[0]
    player_y = state.game_variables[1]
    player_angle = state.game_variables[2]

    # Build map of angles
    angles = np.zeros((240, 320))
    for i in range(depth_buffer.shape[1]):
        angles[:, 319 - i] = i*(fov/width) + (player_angle - fov/2)

    # Build map of absolute coordinates
    d = depth_buffer * game_unit
    ray_y = player_y + (d * np.sin(np.deg2rad(angles)))
    ray_x = player_x + (d * np.cos(np.deg2rad(angles)))
    ray_xy = np.dstack((ray_x, ray_y))

    return ray_xy


def label_seg_map(state, pos, radius, colour, seg_map=None, weight_map=None, debug=False):
    '''
    Paint labels onto scene observations obtained from the agent.
    '''
    # Initialize seg map and get position from state
    player_x = state.game_variables[0]
    player_y = state.game_variables[1]
    dist = np.linalg.norm([player_x - pos[0], player_y - pos[1]])
    if seg_map is None:
        seg_map = np.zeros((240, 320), dtype=np.uint8)
        weight_map = np.zeros((240, 320), dtype=np.uint8)

    # Compute map of relative coordinates to pos
    view_xy = get_view_xy(state)
    rel_view_xy = np.copy(view_xy)
    rel_view_xy[:, :, 0] = rel_view_xy[:, :, 0] - pos[0]
    rel_view_xy[:, :, 1] = rel_view_xy[:, :, 1] - pos[1]
    view_dist = np.linalg.norm(rel_view_xy, axis=2)

    # Label points with Euclidean distance less than radius from pos
    depth_frame = state.depth_buffer
    weight_map = np.copy(weight_map)
    if colour == 1:
        max_height = int(max((200 - dist*0.3), 100))
        view_dist[:max_height, :] = 9999
        label_pts_max = np.where((view_dist < radius*0.50) & (depth_frame > 5))
        label_pts_med = np.where((view_dist >= radius*0.50) & (view_dist < radius*0.75) & (depth_frame > 5))
        label_pts_min = np.where((view_dist >= radius*0.75) & (view_dist <= radius) & (depth_frame > 5))
    else:
        max_height = int(max((150 - dist*0.6), 20))
        view_dist[:max_height, :] = 9999
        label_pts_max = np.where((view_dist < radius*0.50))
        label_pts_med = np.where((view_dist >= radius*0.50) & (view_dist < radius*0.75))
        label_pts_min = np.where((view_dist >= radius*0.75) & (view_dist <= radius))

    for i in range(len(label_pts_min[0])):
        cv2.circle(seg_map, (label_pts_min[1][i], label_pts_min[0][i]), 1, colour,
                   thickness=-1)
        cv2.circle(weight_map, (label_pts_min[1][i], label_pts_min[0][i]), 1, 2,
                   thickness=-1)
    for i in range(len(label_pts_med[0])):
        cv2.circle(seg_map, (label_pts_med[1][i], label_pts_med[0][i]), 1, colour,
                   thickness=-1)
        cv2.circle(weight_map, (label_pts_med[1][i], label_pts_med[0][i]), 1, 3,
                   thickness=-1)
    for i in range(len(label_pts_max[0])):
        cv2.circle(seg_map, (label_pts_max[1][i], label_pts_max[0][i]), 1, colour,
                   thickness=-1)
        cv2.circle(weight_map, (label_pts_max[1][i], label_pts_max[0][i]), 1, 4,
                   thickness=-1)

    # Visualize intermediate output if debug mode enabled
    if debug:
        vis_xy = np.zeros((240, 320, 3))
        vis_xy[:, :, :2] = view_xy
        _, axarr = plt.subplots(2, 2)
        axarr[0][0].imshow(vis_xy[:, :, 0])
        axarr[0][1].imshow(vis_xy[:, :, 1])
        axarr[1][0].imshow(view_dist)
        axarr[1][1].imshow(seg_map)
        plt.show()

    return seg_map, weight_map


def check_valid_position(game):
    '''
    Checks if the agent is in a valid position to sample from.
    '''
    # Check if agent is capable of moving (not stuck in wall)
    pre_x = game.get_game_variable(vzd.GameVariable.POSITION_X)
    pre_y = game.get_game_variable(vzd.GameVariable.POSITION_Y)
    game.make_action([0, 10])
    dx = game.get_game_variable(vzd.GameVariable.POSITION_X) - pre_x
    dy = game.get_game_variable(vzd.GameVariable.POSITION_Y) - pre_y
    norm = np.linalg.norm([dx, dy])
    if norm < 0.1 or norm > 10:
        return False

    # Check if agent was spawned on ground level
    cur_z = game.get_game_variable(vzd.GameVariable.POSITION_Z)
    if cur_z > 20:
        return False

    return True


def max_uncertainty_path(game, model, fs_map, cur_goal, global_map_size=1024):
    '''
    Plan path to sampling goal that maximizes model uncertainty.
    '''
    # Compute global cost map using model
    global_cost_map, origin = compute_global_cost_map(game, model, fs_map)

    # Compute path to goal that maximizes uncertainty
    _, abs_path = plan_sample_path(game, global_cost_map, cur_goal, origin, global_map_size)
    return abs_path


def compute_global_cost_map(game, model, fs_map, global_map_size=1024):
    '''
    Compute global cost map to use for maximum uncertainty planning.
    '''
    # Get RGBD frame from game
    state = game.get_state()
    rgb_frame = np.rollaxis(state.screen_buffer, 0, 3)
    depth_frame = np.expand_dims(state.depth_buffer, axis=2)
    rgbd_frame = np.concatenate((rgb_frame, depth_frame), axis=2)

    # Compute seg map
    _, conf_map = affordance.segment_view(model, rgbd_frame)
    proj_map, valid_map = affordance.project_conf_map(rgbd_frame, conf_map)

    # Set costs for navigation planning
    cost_map = np.copy(proj_map)
    cost_map[:, :] = 10
    cost_map[np.where(fs_map == 2)] = 3
    cost_map[np.where((proj_map <= -1) & (proj_map >= -5))] = 1
    cost_map[np.where(fs_map == 1)] = 100
    for x in [31, 32, 33]:
        for i in range(3):
            if valid_map[65 - (i + 1), x]:
                cost_map[65 - (i + 1):, x] = 1
                break

    # Set origin as current position
    cur_x, cur_y, _ = game.get_agent_location()
    origin = (cur_x, cur_y)

    # Initialize global map with values from local map
    global_cost_map = np.ones((global_map_size, global_map_size)) * 10
    global_cost_map[512-65:512, 512-33:512+32] = cost_map

    return global_cost_map, origin


def plan_sample_path(game, global_map, cur_goal, origin, global_map_size=1024):
    '''
    Plan shortest path to sample goal.
    '''
    cur_x, cur_y, _ = game.get_agent_location()
    cur_abs_pos = (cur_x, cur_y)
    cur_map_pos = util.abs_to_global_map_pos(game, cur_abs_pos, origin, global_map_size)
    map_goal = util.abs_to_global_map_pos(game, cur_goal, origin, global_map_size)

    # Compute absolute and map path
    abs_path = []
    path = planning.shortest_path_geometric(global_map, cur_map_pos, map_goal)
    for cell in path:
        abs_pos = util.global_map_to_abs_pos(game, cell, origin, global_map_size)
        abs_path.append(abs_pos)

    return path, abs_path


def max_uncertainty_locomotion(game, global_path):
    '''
    Navigate towards sequence of waypoints that maximizes model uncertainty.
    '''
    # Select goal from global path
    cur_goal = global_path[0]

    # Use beeline locomotion to navigate towards goal
    locomotion.navigate_beeline(game, cur_goal)

    # Check if intermediate goal has been reached
    player_x, player_y, _ = game.get_agent_location()
    if math.sqrt(abs(player_x - cur_goal[0])**2 + abs(player_y - cur_goal[1])**2) < 10:
        del global_path[0]


def filter_states(states, rgbd_frames):
    states_x = [state.game_variables[0] for state in states]
    states_y = [state.game_variables[1] for state in states]

    good_idxs = [0]
    for i in range(1, len(states)):
        xy_diff = np.array(states_x[i] - states_x[i - 1], states_y[i] - states_y[i - 1])
        diff_norm = np.linalg.norm(xy_diff)
        if diff_norm > 10:
            good_idxs.append(i)

    states = [states[idx] for idx in good_idxs]
    rgbd_frames = [rgbd_frames[idx] for idx in good_idxs]
    return states, rgbd_frames


def sample(args, game, valid_xy, model=None):
    '''
    Initialize and conduct one full sampling episode.
    '''
    # Start new episode and warp to random goal
    game.new_episode()
    game.send_game_command('iddqd')
    random_xy = random.choice(valid_xy)
    game.send_game_command('warp {} {}'.format(random_xy[0], random_xy[1]))

    # Rotate to random angle and check if position is valid
    locomotion.rotate_random_angle(game)
    valid_sample = check_valid_position(game)
    if not valid_sample:
        return -1, None, None, None

    # Compute and select goal in visibility
    if args.freespace_only:
        _, fs_map = util.compute_map(game)
    else:
        fs_map = np.load('./common/visible.npy')
    cur_goal, _ = pick_goal(game, fs_map, {})
    if cur_goal is None:
        return -1, None, None, None

    # Compute affordance map if model specified
    if model is not None:
        global_path = max_uncertainty_path(game, model, fs_map, cur_goal)
        if len(global_path) == 0:
            return -1, None, None, None
        max_steps = max(200, len(global_path) * 50)
    else:
        max_steps = 200

    # Build RGBD input
    label_states = []
    rgbd_frames = []
    original_state = game.get_state()
    label_states.append(original_state)
    rgb_frame = np.rollaxis(original_state.screen_buffer, 0, 3)
    depth_frame = np.expand_dims(original_state.depth_buffer, axis=2)
    rgbd_frame = np.concatenate((rgb_frame, depth_frame), axis=2)
    rgbd_frames.append(rgbd_frame)

    # Attempt to navigate towards goal using beeline policy
    steps_taken = 0
    waypoints = []
    game.send_game_command('iddqd')
    prev_health = game.get_game_variable(vzd.GameVariable.HEALTH)

    while util.dist_to(game, cur_goal) > 30 and steps_taken < max_steps:
        # Take action
        if model is not None:
            max_uncertainty_locomotion(game, global_path)
        else:
            locomotion.navigate_beeline(game, cur_goal)

        # Check if episode is terminated
        if game.get_state() is None:
            return -1, None, None, None

        cur_x, cur_y, _ = game.get_agent_location()
        cur_health = game.get_game_variable(vzd.GameVariable.HEALTH)
        health_diff = prev_health - cur_health

        # Record waypoints
        if (steps_taken % 5) == 0:
            if len(waypoints) == 0 or np.linalg.norm([waypoints[-1][0] - cur_x,
                                                      waypoints[-1][1] - cur_y]) > 5:
                waypoints.append((cur_x, cur_y))
                if len(waypoints) % 4 == 0:
                    cur_state = game.get_state()
                    label_states.append(cur_state)
                    cur_rgb_frame = np.rollaxis(cur_state.screen_buffer, 0, 3)
                    cur_depth_frame = np.expand_dims(cur_state.depth_buffer, axis=2)
                    cur_rgbd_frame = np.concatenate((cur_rgb_frame, cur_depth_frame), axis=2)
                    rgbd_frames.append(cur_rgbd_frame)

        # Check if episode should terminate from damage
        if health_diff > 15:
            # STATUS: Took large amount of damage (probably environmental hazard)
            seg_maps = [None for label_state in label_states]
            weight_maps = [np.zeros((240, 320), dtype=np.uint8) for label_state in label_states]
            for idx, label_state in enumerate(label_states):
                for waypoint in waypoints[:-5]:
                    seg_maps[idx], weight_map = label_seg_map(label_state, waypoint, 25, 1, seg_maps[idx], weight_maps[idx])
                    weight_maps[idx] = np.maximum(weight_map, weight_maps[idx])
                for waypoint in waypoints[-5:]:
                    seg_maps[idx], weight_map = label_seg_map(label_state, waypoint, 50, 2, seg_maps[idx], weight_maps[idx])
                    weight_maps[idx] = np.maximum(weight_map, weight_maps[idx])
                seg_maps[idx], weight_maps[idx] = label_seg_map(label_state, (cur_x, cur_y), 50, 2, seg_maps[idx], weight_maps[idx])
                weight_maps[idx] = np.maximum(weight_map, weight_maps[idx])
            return 1, rgbd_frames, seg_maps, weight_maps
        elif util.check_monster_collision(game):
            # STATUS: Collided with dynamic hazard (probably monster)
            seg_maps = [None for label_state in label_states]
            weight_maps = [np.zeros((240, 320), dtype=np.uint8) for label_state in label_states]
            for idx, label_state in enumerate(label_states):
                for waypoint in waypoints:
                    seg_maps[idx], weight_map = label_seg_map(label_state, waypoint, 25, 1, seg_maps[idx], weight_maps[idx])
                    weight_maps[idx] = np.maximum(weight_map, weight_maps[idx])
                seg_maps[idx], weight_map = label_seg_map(label_state, (cur_x, cur_y), 40, 2, seg_maps[idx], weight_maps[idx])
                weight_maps[idx] = np.maximum(weight_map, weight_maps[idx])
            return 2, rgbd_frames, seg_maps, weight_maps

        prev_health = cur_health
        steps_taken += 1

    # Invalid sample if episode terminates after 0 steps
    if steps_taken == 0:
        return -1, None, None, None

    # Successful sample if goal reached under max_steps
    if steps_taken < max_steps:
        # STATUS: Sample success
        seg_maps = [None for label_state in label_states]
        weight_maps = [np.zeros((240, 320), dtype=np.uint8) for label_state in label_states]
        for idx, label_state in enumerate(label_states):
            for waypoint in waypoints:
                seg_maps[idx], weight_map = label_seg_map(label_state, waypoint, 25, 1, seg_maps[idx], weight_maps[idx])
                weight_maps[idx] = np.maximum(weight_map, weight_maps[idx])

        return 0, rgbd_frames, seg_maps, weight_maps
    # Failed sample if more than 200 steps passed without reaching goal
    else:
        # STATUS: Sample failed to reach goal
        # Get rid of label frames that don't go anywhere
        label_states, rgbd_frames = filter_states(label_states, rgbd_frames)

        rgbd_frames = rgbd_frames
        seg_maps = [None for label_state in label_states]
        weight_maps = [np.zeros((240, 320), dtype=np.uint8) for label_state in label_states]
        for idx, label_state in enumerate(label_states):
            for waypoint in waypoints:
                seg_maps[idx], weight_map = label_seg_map(label_state, waypoint, 25, 1, seg_maps[idx], weight_maps[idx])
                weight_maps[idx] = np.maximum(weight_map, weight_maps[idx])
            seg_maps[idx], weight_map = label_seg_map(label_state, (cur_x, cur_y), 25, 2, seg_maps[idx], weight_maps[idx])
            weight_maps[idx] = np.maximum(weight_map, weight_maps[idx])

        return 3, rgbd_frames, seg_maps, weight_maps
