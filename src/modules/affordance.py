import cv2
import math
import numpy as np


def segment_view(model, rgbd_frame, threshold=0.5):
    '''
    Use trained segmentation model to predict pixel-wise navigability.
    '''
    input = np.zeros((1, 256, 320, 4))
    input[0, 8:-8, :, :] = rgbd_frame

    conf_map = model.predict(input)[0, 8:-8, :, :]
    conf_map[:120, :, :] = 0

    seg_map = (conf_map > threshold).astype(np.uint8)
    return seg_map, conf_map


def project_seg_map(rgbd_frame, seg_map):
    '''
    Project segmented first-person view of navigability into top-down 2D space.
    '''
    width = 320.0
    map_size = 64
    canvas_size = 2*map_size + 1
    map_scale = 8
    offset = 315
    fov = 90.0
    game_unit = 100.0/14

    # Mask seg map edges
    masked_seg_map = np.zeros_like(seg_map)
    masked_seg_map[:, 80:240, 0] = seg_map[:, 80:240, 0]
    seg_map = masked_seg_map

    proj_map = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    mon_pixels = np.nonzero(seg_map)
    for i in range(len(mon_pixels[0])):
        map_x = mon_pixels[1][i]
        map_y = mon_pixels[0][i]

        map_depth = rgbd_frame[map_y, map_x, -1]
        d = (map_depth * game_unit) / float(map_scale)
        theta = ((float(map_x) / width) * fov)
        ray_y = int(round(d * math.sin(math.radians(offset - theta))))
        ray_x = -int(round(d * math.cos(math.radians(offset - theta))))
        proj_y = ray_y + map_size
        proj_x = ray_x + map_size
        if proj_y >= 0 and proj_x >= 0 and proj_y < canvas_size and proj_x < canvas_size:
            cv2.circle(proj_map, (proj_x, proj_y), 1, 1, thickness=-1)

    proj_map = proj_map[:map_size + 1, map_size - 32:map_size + 32 + 1]
    return proj_map


def project_conf_map(rgbd_frame, conf_map, log_scale=True):
    '''
    Project segmented first-person view of confidence values into top-down 2D space.
    '''
    # Set hyper-parameters
    width = 320.0
    map_size = 64
    canvas_size = 2*map_size + 1
    map_scale = 8
    offset = 315
    fov = 90.0
    game_unit = 100.0/14

    # Mask confidence map edges
    masked_conf_map = np.zeros_like(conf_map)
    masked_conf_map[:, 80:240, 0] = conf_map[:, 80:240, 0]
    conf_map = masked_conf_map

    # Initialize projected map
    proj_map = np.zeros((canvas_size, canvas_size))
    count_map = np.zeros((canvas_size, canvas_size))
    proj_pixels = np.where(conf_map > 1e-8)

    # Project non-zero points into 2D space
    for i in range(len(proj_pixels[0])):
        map_x = proj_pixels[1][i]
        map_y = proj_pixels[0][i]

        map_depth = rgbd_frame[map_y, map_x, -1]
        d = (map_depth * game_unit) / float(map_scale)
        theta = ((float(map_x) / width) * fov)
        ray_y = int(round(d * math.sin(math.radians(offset - theta))))
        ray_x = -int(round(d * math.cos(math.radians(offset - theta))))
        proj_y = ray_y + map_size
        proj_x = ray_x + map_size

        # Ignore invalid projected coordinates
        if proj_y < 0 or proj_x < 0:
            continue

        if log_scale:
            proj_map[proj_y, proj_x] += np.log10(conf_map[map_y, map_x])
        else:
            proj_map[proj_y, proj_x] += conf_map[map_y, map_x]
        count_map[proj_y, proj_x] += 1

    # Build map of valid cells for which we have affordance information
    valid_map = np.copy(count_map)[:map_size + 1, map_size - 32:map_size + 32 + 1]
    valid_map[np.where(valid_map > 1)] = 1

    # Normalize projection values in cells with multiple counts
    count_map[np.where(count_map == 0)] = 1
    proj_map = proj_map / count_map
    proj_map = proj_map[:map_size + 1, map_size - 32:map_size + 32 + 1]

    # proj_map = gaussian_filter(proj_map, sigma=0.5)
    return proj_map, valid_map
