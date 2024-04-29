#!/bin/bash
echo 'Submitting SBATCH jobs...'

################### Define a few global run parameters #######################
time="24:00:00"
ram="32G" # Amount of RAM
vram="32gb" # Amount of GPU memory


# Modify this according to your own directory structure!
project_path="/home/mila/c/cristian-dragos.manta/adversarial-ml/classifier-recovery"
python_env_name="dem"
##############################################################################

# Boilerplate
job_setup () {
    # Job resources requirements
    echo "#!/bin/bash" >> temprun.sh
    echo "#SBATCH --partition=long"  >> temprun.sh
    echo "#SBATCH --cpus-per-task=2" >> temprun.sh
    echo "#SBATCH --gres=gpu:$vram:1" >> temprun.sh
    echo "#SBATCH --mem=$ram" >> temprun.sh
    echo "#SBATCH --time=$time " >>  temprun.sh
    echo "#SBATCH -o $project_path/slurm-%j.out" >> temprun.sh

    # Environment setup
    echo "module purge" >> temprun.sh
    echo "module load cuda/11.8" >> temprun.sh
    echo "MAMBA_EXE='/home/mila/c/cristian-dragos.manta/.local/bin/micromamba'" >> temprun.sh
    echo 'MAMBA_ROOT_PREFIX='/home/mila/c/cristian-dragos.manta/micromamba'' >> temprun.sh
    echo '__mamba_setup="$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"' >> temprun.sh
    echo 'eval "$__mamba_setup"' >> temprun.sh
    echo "micromamba activate $python_env_name" >> temprun.sh
}

job_setup

echo "python DEM/dem/train.py experiment=classifier_idem" >> temprun.sh
sbatch temprun.sh
rm temprun.sh