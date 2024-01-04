
# See
# https://github.com/instadeepai/og-marl
# https://sites.google.com/view/og-marl (download old project files from here)

# Conda Setup
conda init bash
conda create --name conda39-jumanji python=3.9

source activate base	
conda deactivate
conda activate conda39-jumanji
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

# Install dependencies
bash install_environments/flatland.sh
bash install_environments/smacv2.sh
pip install dm-sonnet
pip install cpprb
pip install opencv-python
pip install https://github.com/instadeepai/Mava/archive/refs/tags/0.1.2.zip
pip3 install numpy --upgrade


# Download datasets (remember to put __init__.py file)
python3 -m examples.download_dataset
or
download from this link -> https://sites.google.com/view/og-marl

# Run examples
python3 -m examples.tf2.online.qmix_smacv2
python3 -m examples.profile_datasets.profile_flatland
python3 -m examples.baselines.benchmark_smac --algo_name=qmix --dataset_quality=Good --env_name=3m
python3 -m examples.baselines.run_pistonball --algo_name=qmix --dataset_quality=Good --env_name=3m

# Start Tensorboard
tensorboard --logdir=./ --port=8080


