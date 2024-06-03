#!/usr/bin/env bash
#SBATCH --job-name=train-valle-nar            # Job name
#SBATCH --output result.out         ## filename of the output
#SBATCH --nodes=1                   ## Run all processes on a single node	
#SBATCH -w, --nodelist=node03             ## Request the pointed node  
#SBATCH --ntasks=8                  ## number of processes = 20
#SBATCH --cpus-per-task=1           ## Number of CPU cores allocated to each process
#SBATCH --partition=Project         ## Partition name: Project or Debug (Debug is default)
export PYTHONPATH="./"

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated      = $SLURM_NTASKS"
echo "Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"
echo "Number of CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo ""
echo "Running script... "
 
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
 

######## Build Experiment Environment ###########
exp_dir="/nfsmnt/qiujunwen/AmphionVALLEv2/egs/tts/valle_gpt_simple"
echo exp_dir
work_dir="/nfsmnt/qiujunwen/AmphionVALLEv2/"
echo work_dir
echo PATH


export WORK_DIR=$work_dir
export PYTHONPATH=$work_dir
export PYTHONIOENCODING=UTF-8
 
cd $work_dir/modules/monotonic_align
mkdir -p monotonic_align
python setup.py build_ext --inplace
cd $work_dir

if [ -z "$exp_config" ]; then
    exp_config="${exp_dir}"/exp_nar_libritts.json
fi
echo "Exprimental Configuration File: $exp_config"

exp_name="nar_libritts"

port=17004


######## Train Model ###########
echo "Experimental Name: $exp_name"
CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port $port "${work_dir}"/bins/tts/train.py --config $exp_config --exp_name $exp_name --log_level debug --seed 1234 \
    #--resume \
    #--resume_type "resume" \
    #--resume_from_ckpt_path "/nfsmnt/qiujunwen/ckpt/valle_gpt_simple/ar_libritts_dev_clean/checkpoint/epoch-0373_step-1158000_loss-0.000001"


# uncomment the "resume" part to automatically resume from the last-time checkpoint