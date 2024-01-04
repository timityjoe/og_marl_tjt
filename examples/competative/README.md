# Collecting Offline Data on a Competetive Environment

In the script `examples/competetive/simple_adversary.py` we demonstrate how to use the OG-MARL data logger to collect offline experience in a competetive scenario using independent Q-Learner (IQL) agents.

We use the [Simple Adversary](https://pettingzoo.farama.org/environments/mpe/simple_adversary/) environment from PettingZoo. To run the code you will need to install OG-MARL with baseline and dataset dependencies. Additionally you will need to install PettingZoo. Please see the main README.

In the code we first create a simple environment wrapper to make in compatible with the data logger and the IQL system. After defining the environment wrapper, we then instantiate the wrapped environment, the data logger and the system. Running the script will then start training the IQL system and log the offline data to the directory `./offline_env_logs`. 