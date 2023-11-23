#!/bin/bash



# Define the total number of tiles and tiles per job
TOTAL_TILES=208
TILES_PER_JOB=8

# Calculate the number of jobs needed
NUM_JOBS=$((TOTAL_TILES / TILES_PER_JOB))

# Loop over the number of jobs
for (( JOB=1; JOB<=NUM_JOBS; JOB++ ))
do
    # Calculate the start and end tile for the current job
    START_TILE=$(( (JOB - 1) * TILES_PER_JOB ))
    END_TILE=$(( JOB * TILES_PER_JOB ))

    # Modify the config file or create a new one for the current batch
    # This depends on how your config file is structured and how it's read by your Python code
    # For example, you can use 'sed' to modify the tile range in the config file
    # sed -i "s/original_range/$START_TILE-$END_TILE/" config/config_measure_template_moments_only.yaml

    # Run the MPI job for the current batch of tiles
    echo "Processing tiles from $START_TILE to $END_TILE"

    srun --nodes=4 --tasks-per-node=2 python ./run_bfd.py --config ./config/config_measure_target_moments_only.yaml --output_label _run1 --start_tile $START_TILE --end_tile $END_TILE

    # Check if the run was successful, handle errors if needed
    # ...

    # Optionally, wait a bit before starting the next job to free up resources
    sleep 10
done