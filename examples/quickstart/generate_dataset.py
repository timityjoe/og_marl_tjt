import numpy as np

"""PART 1: This is a demonstration of how to generate a new offline MARL dataset
using OG-MARL. For this demonstration we will make an environment with
two independent CartPole instances. We will then generate a dataset using
random policy.

We defined the DoubleCartPole environment in `examples/quickstart/double_cartpole.py`
"""

from .double_cartpole import DoubleCartPole
from loguru import logger
logger.remove()
# logger.add(sys.stdout, level="INFO")
# logger.add(sys.stdout, level="SUCCESS")
# logger.add(sys.stdout, level="WARNING")

logger.info("1) Create DoubleCartPole env")
# marl_env = DoubleCartPole()
double_cart_pole_env = DoubleCartPole()

"""PART 2: In Order to record experiences in our new environment we must wrap
it in the offline utility provided by OG-MARL."""

from og_marl.offline_tools.offline_environment_logger import MAOfflineEnvironmentSequenceLogger

logger.info("2) Create MAOfflineEnvironmentSequenceLogger")
marl_env = MAOfflineEnvironmentSequenceLogger(
    environment=double_cart_pole_env,
    sequence_length=5,
    period=5,
    logdir="./datasets/double_cartpole",
    min_sequences_per_file=250
)

"""PART 3: Next we need an executor to coordinate action selection for all
of the agents in the environment. Our executor will just choose random actions
in this tutorial."""

from og_marl.systems.executor_base import ExecutorBase

class RandomExecutor(ExecutorBase):

    def __init__(self, agents):

        super().__init__(
            agents=agents,
            variable_client=None
        )

    def observe_first(self, timestep, extras={}):
        # Nothing to do
        return

    def observe(self, actions, next_timestep, next_extras):
        # Nothing to do
        return

    def select_actions(self, observations):
        actions = {}
        for agent in self._agents:
            actions[agent] = np.random.randint(2)
        return actions

executor = RandomExecutor(marl_env.agents)

"""PART 4: Now that we have setup an executor and environment, we can
put it all together using the environmentloop in OG-MARL and start 
generating a dataset."""

from og_marl.environment_loop import EnvironmentLoop

env_loop = EnvironmentLoop(
    marl_env,
    executor,
    logger=None,
    )


NUM_EPISODES = 3000
logger.info("3) Running loops for NUM_EPISODES:{NUM_EPISODES} times")
for e in range(NUM_EPISODES):
    # render_env = True
    render_env = False
    env_loop.run_episode(render_env)
    if e % 100 == 0:
        print(f"{e} Episodes Done.")

"""PART 5: Your dataset should appear in the directory `./datasets`
Now move on to the `example/train_offline_algos.`py` script for 
PART 6.
"""