
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
pip install docker_pycreds appdirs
pip install sentry-sdk==1.0.0 setproctitle
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
python examples/<backend>/main.py --system=<system_name> --env=<env_name> --scenario=<scenario_name>
python examples/tf2/main.py --system=idrqn --env=flatland --scenario=3_trains
python3 -m examples.tf2.main --system=idrqn --env=flatland --scenario=5_trains


# Start Tensorboard
tensorboard --logdir=./ --port=8080


