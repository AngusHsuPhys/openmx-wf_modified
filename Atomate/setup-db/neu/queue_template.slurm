#!/bin/bash -l

#SBATCH --nodes=$${nodes}
#SBATCH --ntasks=$${ntasks}
#SBATCH --ntasks-per-node=$${ntasks_per_node}
#SBATCH --cpus-per-task=$${cpus_per_task}
#SBATCH --mem=$${mem}
#SBATCH --gres=$${gres}
#SBATCH --qos=$${qos}
#SBATCH --time=$${walltime}
#SBATCH --partition=$${queue}
#SBATCH --account=$${account}
#SBATCH --job-name=$${job_name}
#SBATCH --license=$${license}
#SBATCH --output=$${job_name}-%j.out
#SBATCH --error=$${job_name}-%j.error
#SBATCH --constraint=$${constraint}

ssh -Nf db-sinica

export PATH=$HOME/openmx3.9_neu/source:$PATH
module load intel-oneapi/2021.1_u9
source /shared/centos7/intel/oneapi/2021.1_u9-base/setvars.sh


source /home/j.tsai/software/anaconda3/etc/profile.d/conda.sh
conda activate openmx



$${pre_rocket}
cd $${launch_dir}
$${rocket_launch}
$${post_rocket}

# CommonAdapter (SLURM) completed writing Template