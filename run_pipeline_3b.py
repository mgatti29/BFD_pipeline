import BFD_pipeline

conf=BFD_pipeline.read_config('config_run_3b.yaml')
#conf=BFD_pipeline.read_config('config_run_multitiles.yaml')

BFD_pipeline.BFD_pipeline(conf)