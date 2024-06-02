#!/bin/sh
#SBATCH --job-name=your_job_name    ## Job name
#SBATCH --output result.out         ## filename of the output
#SBATCH --nodes=1                   ## Run all processes on a single node	
#SBATCH --ntasks=8                  ## number of processes = 20
#SBATCH --cpus-per-task=1           ## Number of CPU cores allocated to each process
#SBATCH --partition=Project         ## Partition name: Project or Debug (Debug is default)
python ./models/tts/valle_gpt_simple/valle_nar.py