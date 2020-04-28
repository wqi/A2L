import argparse
import pickle as pkl
import time

import common.util as util
from preprocess.random_explore import explore_map_random_policy


'''
This script is used to generate beacons located in freespace from any compatible Doom map.

The locations associated with these beacons can be used as agent spawn locations for
self-supervised sampling.

Configuration is performed via command line arguments specified below.
'''


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Paths
    parser.add_argument('--wad-dir', type=str, default='../data/maps/test',
                        help='Path to dir containing map files')
    parser.add_argument('--save-dir', type=str, required=True,
                        help='Path to save PKLs with generated beacons')

    # Beacon Options
    parser.add_argument('--iters', type=int, default=5,
                        help='Number of iterations to explore for, accumulating beacons from each')

    args = parser.parse_args()
    return args


def filter_graph(nodes, edges):
    nodes_with_edges = {}
    for e in edges:
        nodes_with_edges[e[0]] = True
        nodes_with_edges[e[1]] = True

    for n in list(nodes.keys()):
        if n not in nodes_with_edges:
            nodes.pop(n)


def generate_beacons(args):
    default_cfg_path = '../data/configs/default.cfg'
    wad_ids = util.get_sorted_wad_ids(args.wad_dir)

    # Generate beacons for each map in dir
    for idx, wad_id in enumerate(wad_ids):
        start = time.time()

        # Explore map with random policy and build beacon connectivity graph
        nodes = {}
        edges = {}
        for i in range(args.iters):
            explore_map_random_policy(default_cfg_path, args.wad_dir, wad_id,
                                      nodes=nodes, edges=edges)
            print('{} nodes accumulated...'.format(len(nodes)))
        filter_graph(nodes, edges)

        # Dump accumulated beacon nodes to disk as PKL
        print('{} nodes after filtering...'.format(len(nodes)))
        with open('./{}.pkl'.format(wad_id), 'wb') as handle:
            pkl.dump(nodes, handle, protocol=pkl.HIGHEST_PROTOCOL)

        # Report statistics for current map
        end = time.time()
        elapsed_time = end - start
        print('Finished exploring map {} for {} iterations in {}s'.format(
            idx, args.iters, elapsed_time
        ))


if __name__ == "__main__":
    args = parse_arguments()
    generate_beacons(args)
