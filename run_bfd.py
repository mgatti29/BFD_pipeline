#!/usr/bin/env python
import BFD_pipeline
import argparse
from galsim.utilities import Profile

#https://stackoverflow.com/questions/20063/whats-the-best-way-to-parse-command-line-arguments
# Instantiate the parser
parser = argparse.ArgumentParser(description='Optional app description')

parser.add_argument('--config', type=str,
                    help='required config file')

parser.add_argument('--tiles', nargs='+', type=str, help='required config file')
parser.add_argument('--output_label', type=str, help='required config file')
parser.add_argument('--start_tile', type=str, default='0')
parser.add_argument('--end_tile', type=str, default='0')



args = parser.parse_args()
conf=BFD_pipeline.read_config(args.config)


args.start_tile = int(args.start_tile)
args.end_tile = int(args.end_tile)

if 'measure_moments_targets' in conf.keys():
    if args.tiles is not None:
        conf['measure_moments_targets']['tiles'] = args.tiles

    if len(conf['measure_moments_targets']['tiles'])>1:
        if args.start_tile != args.end_tile:
            conf['measure_moments_targets']['tiles'] = conf['measure_moments_targets']['tiles'][args.start_tile:args.end_tile]

    if args.output_label is not None:
        conf['measure_moments_targets']['output_label'] = args.output_label

#with Profile(filename="bfd.prof"):
#    #gprof2dot -f pstats bfd.prof | dot -Tpng -o output.png
#    BFD_pipeline.BFD_pipeline(conf)
#
BFD_pipeline.BFD_pipeline(conf)