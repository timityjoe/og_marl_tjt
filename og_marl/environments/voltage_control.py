"""MAPDN Environment Wrapper."""
from typing import Dict, List
import numpy as np
from og_marl.environments.base import BaseEnvironment
from gymnasium.spaces import Box
from var_voltage_control.voltage_control_env import VoltageControl 


class VoltageControlEnv(BaseEnvironment):
    """Environment wrapper for MAPDN environment."""

    def __init__(self):
        """Constructor for VoltageControl."""
        self._environment = VoltageControl()
        self.possible_agents = [f"agent_{id}" for id in range(self._environment.get_num_of_agents())]
        self.num_agents = len(self.possible_agents)
        self._num_actions = self._environment.get_total_actions()

        self.action_spaces = {agent: Box(-1, 1, (self._num_actions,), "float32") for agent in self.possible_agents}
        self.observation_spaces = {agent: Box(-np.inf, np.inf, (50,), "float32") for agent in self.possible_agents}
        self.info_spec = {
            "state": np.zeros((144,), "float32"),
            "legals": {agent: np.zeros((1,), "float32") for agent in self.possible_agents} # placeholder
        }

    def reset(self):
        """Resets the env.

        Returns:
            dm_env.TimeStep: dm timestep.
        """
        # Reset the environment
        observations, state = self._environment.reset()

        # Global state
        info = {
            "state": state.astype("float32"),
            "legals": np.zeros((1,), "float32") # placeholder
        }

        self._done = False

        # Convert observations
        observations = self._convert_observations(observations, self._done)

        return observations, info

    def step(self, actions: Dict[str, np.ndarray]):
        """Steps in env.
        """
        actions = self._preprocess_actions(actions)

        # Step the environment
        reward, done, _ = self._environment.step(
            actions
        )

        rewards = {}
        for agent in self.possible_agents:
                rewards[agent] = np.array(reward, "float32")

        # Set done flag
        self._done = done

        next_observations = self._environment.get_obs()

        next_observations = self._convert_observations(next_observations, self._done)

        state = self._environment.get_state().astype("float32")

        # Global state
        info = {
            "state": state,
            "legals": np.zeros((1,), "float32") # placeholder
        }

        terminals = {agent: done for agent in self.possible_agents}
        truncations = {agent: False for agent in self.possible_agents}

        return next_observations, rewards, terminals, truncations, info

    def _preprocess_actions(self, actions):
        concat_action = []
        for agent in self.possible_agents:
            concat_action.append(actions[agent])
        concat_action = np.concatenate(concat_action)
        return concat_action

    def _convert_observations(
        self, observations: List, done: bool
    ):
        """Convert observation so it's dm_env compatible.

        Args:
            observes (Dict[str, np.ndarray]): observations per agent.
            dones (Dict[str, bool]): dones per agent.

        Returns:
            types.Observation: dm compatible observations.
        """
        dict_observations = {}
        for i, agent in enumerate(self.possible_agents):
            obs = np.array(observations[i], "float32")
            dict_observations[agent] = obs
        return dict_observations