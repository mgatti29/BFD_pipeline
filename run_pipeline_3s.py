import BFD_pipeline
import argparse

#https://stackoverflow.com/questions/20063/whats-the-best-way-to-parse-command-line-arguments
# Instantiate the parser
parser = argparse.ArgumentParser(description='Optional app description')

parser.add_argument('config', type=str,
                    help='required config file')



args = parser.parse_args()


conf=BFD_pipeline.read_config(args.config)
BFD_pipeline.BFD_pipeline(conf)
