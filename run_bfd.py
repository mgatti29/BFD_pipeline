#!/usr/bin/env python
import BFD_pipeline
import argparse

#https://stackoverflow.com/questions/20063/whats-the-best-way-to-parse-command-line-arguments
# Instantiate the parser
parser = argparse.ArgumentParser(description='Optional app description')

parser.add_argument('--config', type=str,
                    help='required config file')

parser.add_argument('--tiles', nargs='+', type=str, help='required config file')
parser.add_argument('--output_label', type=str, help='required config file')


args = parser.parse_args()
conf=BFD_pipeline.read_config(args.config)


if args.tiles is not None:
    conf['measure_moments_targets']['tiles'] = [args.tiles]

if args.output_label is not None:
    conf['measure_moments_targets']['output_label'] = args.output_label


BFD_pipeline.BFD_pipeline(conf)
