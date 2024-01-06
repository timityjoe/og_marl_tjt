
# See
# https://github.com/instadeepai/og-marl
# https://sites.google.com/view/og-marl (download old project files from here)

# Conda Setup
conda init bash
conda create --name conda39-ogmarl-old python=3.9

source activate base	
conda deactivate
conda activate conda39-ogmarl-old
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/


# Install dependencies
bash install_environments/flatland.sh
bash install_environments/smacv2.sh
pip install dm-sonnet
pip install cpprb
pip install opencv-python
pip install https://github.com/instadeepai/Mava/archive/refs/tags/0.1.2.zip
pip3 install numpy --upgrade
pip install --upgrade pip


# Download datasets (remember to put __init__.py file)
python3 -m examples.download_dataset
or
download from this link -> https://sites.google.com/view/og-marl


# Run examples
python3 -m examples.tf2.online.qmix_smacv2
python3 -m examples.profile_datasets.profile_flatland
python3 -m examples.baselines.benchmark_smac --algo_name=qmix --dataset_quality=Good --env_name=3m
# algo_name options are [bc, itd3bc, itd3cql, itd3, omar]
python3 -m examples.baselines.run_pistonball --algo_name=bc --dataset_quality=Good --env_name=pistonball_good-1_13_0.1


# Run quickstart examples
python3 -m examples.quickstart.generate_dataset
python3 -m examples.quickstart.train_offline_algo --algo_name=maicq
python3 -m examples.quickstart.train_offline_algo --algo_name=qmix+bcq


# Start Tensorboard
cd logs/tensorboard
tensorboard --logdir=./ --port=8080


