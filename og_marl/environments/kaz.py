from pettingzoo.butterfly import knights_archers_zombies_v10
from og_marl.environments.base import OLT
from og_marl.environments.pettingzoo_base import PettingZooBase
import numpy as np
from dm_env import specs

class KAZ(PettingZooBase):

    def __init__(
        self, render=False
    ):
        if render==True:
            render="human"
        else:
            render = None

        self._environment = knights_archers_zombies_v10.parallel_env(max_cycles=900, num_archers=1, num_knights=1, render_mode=render)

        self.num_actions = 5
        self.num_agents = 2

        self._agents = self._environment.possible_agents
        self._reset_next_step = True
        self._done = False
        self.environment_label = "pettingzoo/kaz"

    def _flatten(self, obs):
        return np.reshape(obs, (5*24,))

    def _create_state_representation(self, observations):

        observations_list = []
        for agent in self._agents:
            agent_obs = self._flatten(observations[agent])

            observations_list.append(agent_obs)

        state = np.concatenate(
            observations_list, axis=-1
        )

        return state

    def _add_zero_obs_for_missing_agent(self, observations):
        for agent in self._agents:
            if agent not in observations:
                observations[agent] = np.zeros((120,), "float32")
        return observations

    def _convert_observations(
        self, observations, done: bool
    ):
        """Convert SMAC observation so it's dm_env compatible.

        Args:
            observes (Dict[str, np.ndarray]): observations per agent.
            dones (Dict[str, bool]): dones per agent.

        Returns:
            types.Observation: dm compatible observations.
        """
        olt_observations = {}
        for i, agent in enumerate(self._agents):

            agent_obs = self._flatten(observations[agent]).astype("float32")
            if np.sum(agent_obs) == 0:
                legals = np.array([0,0,0,0,1], "float32")
            else:
                legals = np.ones((5,), "float32")

            olt_observations[agent] = OLT(
                observation=agent_obs,
                legal_actions=legals,
                terminal=np.asarray(done, dtype="float32"),
            )

        return olt_observations

    def extra_spec(self):
        """Function returns extra spec (format) of the env.

        Returns:
            Dict[str, specs.BoundedArray]: extra spec.
        """
        state_spec = {"s_t": np.zeros((240,), "float32")}  # four stacked frames

        return state_spec

    def observation_spec(self):
        """Observation spec.

        Returns:
            types.Observation: spec for environment.
        """
        observation_specs = {}
        for agent in self._agents:

            obs = np.zeros((120,), "float32")

            observation_specs[agent] = OLT(
                observation=obs,
                legal_actions=np.ones((5,), "float32"),
                terminal=np.asarray(True, "float32"),
            )

        return observation_specs

    def action_spec(
        self,
    ):
        """Action spec.

        Returns:
            spec for actions.
        """
        action_specs = {}
        for agent in self._agents:
            action_specs[agent] = specs.DiscreteArray(
                num_values=5, dtype="int64"  # three actions
            )
        return action_specs