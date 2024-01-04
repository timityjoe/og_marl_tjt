"""Wraper for Simple adversary."""
from typing import Dict, List, Union

from datetime import datetime
from dm_env import specs
import numpy as np
import functools
from og_marl.environments.simple_adversary import SimpleAdversary
from mava.utils.loggers import logger_utils
import sonnet as snt
import tensorflow as tf
from og_marl.environments.base import OLT
from og_marl.environments.pettingzoo_base import PettingZooBase
from og_marl.utils.loggers import WandbLogger, WandbSweppLogger
from og_marl.offline_tools import OfflineLogger
from og_marl.systems.iql.system_builder import IQLSystemBuilder
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               

###############
### RUN EXP ###
###############

# Instantiate Competetive Environment
env_factory = SimpleAdversary

# Loggers
logger_factory = functools.partial(
    logger_utils.make_logger,
    directory="logs",
    to_terminal=True,
    to_tensorboard=True,
    time_stamp=str(datetime.now()),
    time_delta=1,  # log every 1 sec
    external_logger= None,
)

# Q-Network
q_network=snt.DeepRNN(
    [
        snt.Linear(128),
        tf.nn.relu,
        snt.GRU(64),
        tf.nn.relu,
        snt.Linear(5),
    ]
)

# Make system of independent Q-learners
system = IQLSystemBuilder(env_factory, logger_factory, q_network, record_evaluator_every=2, samples_per_insert=4, add_agent_id_to_obs=True, offline_environment_logging=True)

# Collect competetive data
system.run_in_parallel()