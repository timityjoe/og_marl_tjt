
# See
# https://github.com/instadeepai/og-marl
# https://sites.google.com/view/og-marl (download old project files from here)

# FAQ
"TypeError: The `filenames` argument must contain `tf.string` elements. Got `tf.float32` elements."
- Dataset path in following manner: /datasets/flatland/5_trains/Replay

# Conda Setup
conda init bash
conda create --name conda39-jumanji python=3.9

source activate base	
conda deactivate
conda activate conda39-jumanji
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

# Install dependencies
bash install_environments/flatland.sh
bash install_environments/smacv1.sh
bash install_environments/smacv2.sh
pip install https://github.com/instadeepai/Mava/archive/refs/tags/0.1.2.zip
pip3 install numpy --upgrade


# Download datasets (remember to put __init__.py file)
python3 -m examples.download_dataset
or
download from this link -> https://sites.google.com/view/og-marl


# Run examples (og_marl_old)
python3 -m examples.tf2.online.qmix_smacv2
python3 -m examples.profile_datasets.profile_flatland
python3 -m examples.baselines.benchmark_smac --algo_name=qmix --dataset_quality=Good --env_name=3m
python3 -m examples.baselines.run_pistonball --algo_name=qmix --dataset_quality=Good --env_name=3m


# Run examples (og_marl_new)
# See https://github.com/instadeepai/og-marl
# Dataset path in following manner: /datasets/flatland/5_trains/Replay
python examples/<backend>/main.py --system=<system_name> --env=<env_name> --scenario=<scenario_name>
python3 -m examples.tf2.main --system=idrqn --env=flatland --scenario=5_trains
python3 -m examples.tf2.main --system=idrqn --env=smac_v1 --scenario=3m
python3 -m examples.tf2.main --system=idrqn --env=smac_v1 --scenario=8m


# Start Tensorboard
tensorboard --logdir=./ --port=8080


