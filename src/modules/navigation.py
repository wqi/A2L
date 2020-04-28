import cv2
import numpy as np
import vizdoom as vzd

import common.util as util
import modules.affordance as affordance
import modules.locomotion as locomotion
import modules.planning as planning


def visualize(game, merged_map, cur_goal, abs_path, show=False):
    '''
    Visualize local scene representation.
    '''
    map_rgb = cv2.cvtColor(merged_map * 127, cv2.COLOR_GRAY2RGB)

    if abs_path:
        for waypoint in abs_path:
            map_point = util.abs_to_map_pos(game, waypoint)
            cv2.circle(map_rgb, (map_point[1], map_point[0]), 0, (255, 0, 0), thickness=-1)

    if cur_goal:
        map_goal = util.abs_to_map_pos(game, cur_goal)
        cv2.circle(map_rgb, (map_goal[1], map_goal[0]), 2, (0, 0, 255), thickness=-1)

    map_rgb = cv2.resize(map_rgb, dsize=(240, 240))

    if show:
        cv2.imshow('Projected Map', map_rgb)
        cv2.waitKey(0)

    return map_rgb


def visualize_global(game, global_map, global_map_size, origin, cur_goal, abs_path):
    '''
    Visualize global scene representation.
    '''
    map_rgb = cv2.cvtColor(global_map.astype(np.uint8) * 127, cv2.COLOR_GRAY2RGB)

    if abs_path:
        for waypoint in abs_path:
            map_point = util.abs_to_global_map_pos(game, waypoint, origin, global_map_size)
            cv2.circle(map_rgb, (map_point[1], map_point[0]), 2, (255, 0, 0), thickness=-1)

    # Draw current position
    cur_pos = (game.get_game_variable(vzd.POSITION_X), game.get_game_variable(vzd.POSITION_Y))
    map_cur = util.abs_to_global_map_pos(game, cur_pos, origin, global_map_size)
    cv2.circle(map_rgb, (map_cur[1], map_cur[0]), 3, (0, 255, 0), thickness=-1)

    # Draw goal
    map_goal = util.abs_to_global_map_pos(game, cur_goal, origin, global_map_size)
    cv2.circle(map_rgb, (map_goal[1], map_goal[0]), 3, (0, 0, 255), thickness=-1)
    return map_rgb


def plan_path_global(game, global_map, cur_goal, origin, global_map_size, model=None):
    '''
    Plan path towards goal in global coordinate space.
    '''
    cur_x, cur_y, _ = game.get_agent_location()
    cur_abs_pos = (cur_x, cur_y)
    cur_map_pos = util.abs_to_global_map_pos(game, cur_abs_pos, origin, global_map_size)
    map_goal = util.abs_to_global_map_pos(game, cur_goal, origin, global_map_size)

    # Compute absolute and map path
    if model:
        path = planning.shortest_path_affordance(global_map, cur_map_pos, map_goal)
    else:
        path = planning.shortest_path_geometric(global_map, cur_map_pos, map_goal)

    abs_path = []
    for cell in path:
        abs_pos = util.global_map_to_abs_pos(game, cell, origin, global_map_size)
        abs_path.append(abs_pos)

    return path, abs_path


def merge_maps(proj_map, fs_map, threshold):
    '''
    Merge projected navigability map with geometric map to form affordance map.
    '''
    # Initialize local cost map
    local_cost_map = np.zeros_like(proj_map)

    # Set costs for navigation planning
    local_cost_map[np.where((proj_map >= threshold) & (fs_map == 2))] = 10  # High Cost
    local_cost_map[np.where((proj_map < threshold) & (fs_map == 2))] = 1  # Low Cost

    # Set walls to be non-navigable cost
    local_cost_map[np.where(fs_map == 1)] = 999

    return local_cost_map


def navigate(game, max_steps, final_goal, model=None, vid_path=None,
             localization_noise=0.0):
    '''
    Use geometric or affordance-based maps to navigate agent towards final goal.
    '''
    # Set up video writer
    vid_out = None
    if vid_path is not None:
        vid_out = cv2.VideoWriter(vid_path,
                                  cv2.VideoWriter_fourcc(*'mp4v'),
                                  vzd.DEFAULT_TICRATE // 2, (1120, 480))

    # Initialize variables
    abs_path = []
    path_idx = 0
    global_map_size = 1024
    global_cost_map = np.zeros((global_map_size, global_map_size))
    origin_x, origin_y, _ = game.get_agent_location()
    origin = (origin_x, origin_y)
    total_damage = 0
    total_steps = 0
    replan = False
    full_path = []

    for i in range(max_steps):
        # Get current state and buffer
        state = game.get_state()
        if state is None:
            break

        rgb_frame = np.rollaxis(state.screen_buffer, 0, 3)
        depth_frame = np.expand_dims(state.depth_buffer, axis=2)
        rgbd_frame = np.concatenate((rgb_frame, depth_frame), axis=2)

        # Compute geometric free space map
        _, fs_map = util.compute_map(game)
        fs_map[:, :16] = 0
        fs_map[:, 48:] = 0
        local_cost_map = fs_map

        # Compute and merge projected segmentation map if model specified
        if model:
            threshold = 0.4
            seg_map, conf_map = affordance.segment_view(model, rgbd_frame, threshold)
            proj_conf_map, valid_map = affordance.project_conf_map(rgbd_frame, conf_map, False)

            local_cost_map = merge_maps(proj_conf_map, fs_map, threshold)

        # Update global map used for planning
        update_cells = np.where(local_cost_map > 0)
        for cell in zip(update_cells[0], update_cells[1]):
            abs_pos = util.map_to_abs_pos(game, cell)
            global_pos = util.abs_to_global_map_pos(game, abs_pos, origin, global_map_size)
            cur_val = global_cost_map[global_pos[0], global_pos[1]]
            new_val = local_cost_map[cell[0], cell[1]]

            if model:
                global_cost_map[global_pos[0], global_pos[1]] = (cur_val + new_val) / 2
            else:
                global_cost_map[global_pos[0], global_pos[1]] = new_val

        # Replan path to specified goal
        if (i % 10 == 0) or replan:
            path, abs_path = plan_path_global(game, global_cost_map, final_goal, origin,
                                              global_map_size, model)
            replan = False

        # Visualize merged map, goal, and planned path
        if len(abs_path) > 0:
            path_idx = min(len(abs_path) - 1, 3)
            global_map_rgb = visualize_global(game, global_cost_map, global_map_size, origin,
                                              final_goal, abs_path[path_idx:])
            locomotion.navigate_beeline(game, abs_path[path_idx])
        else:
            global_map_rgb = visualize_global(game, global_cost_map, global_map_size, origin,
                                              final_goal, None)
            game.make_action([10, 0])
            replan = True

        # Add current location to path
        cur_x, cur_y, _ = game.get_agent_location()
        full_path.append((round(cur_x), round(cur_y)))

        # Select next goal if reached
        dist_to_goal = util.dist_to(game, final_goal)
        if dist_to_goal < 30:
            print('Goal reached')
            return {'success': 1, 'steps': total_steps, 'damage': total_damage}, full_path

        # Check for damage
        cur_damage = 100 - game.get_game_variable(vzd.HEALTH)
        if cur_damage > 15:
            total_damage += 20
            print('Total Damage: {}'.format(total_damage))
        if util.check_monster_collision(game):
            total_damage += 4
            print('Total Damage: {}'.format(total_damage))
        game.send_game_command("give health")

        # Write to video if specified
        if vid_out:
            vis_buffer = np.zeros((480, 1120, 3), dtype=np.uint8)
            rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
            vis_buffer[:240, :320, :] = rgb_frame

            if model:
                rgb_frame_alpha = 0.7 * rgb_frame
                seg_map_colour = cv2.applyColorMap(seg_map * 255, cv2.COLORMAP_JET)
                seg_map_alpha = 0.3 * seg_map_colour
                seg_map_combined = rgb_frame_alpha + seg_map_alpha
                vis_buffer[240:, :320, :] = seg_map_combined

                conf_map_int = np.rint(conf_map * 255).astype(np.uint8)
                conf_map_colour = cv2.applyColorMap(conf_map_int, cv2.COLORMAP_JET)
                conf_map_alpha = 0.3 * conf_map_colour
                conf_map_combined = rgb_frame_alpha + conf_map_alpha
                vis_buffer[240:, 320:640, :] = conf_map_combined

            vis_buffer[:, 640:, :] = global_map_rgb[512 - 240: 512 + 240, 512 - 240: 512 + 240]
            vid_out.write(vis_buffer)

        total_steps = i + 1

    # Terminate game and write video to disk
    if vid_out:
        vid_out.release()
    return {'success': 0, 'steps': 1000, 'damage': total_damage}, full_path
