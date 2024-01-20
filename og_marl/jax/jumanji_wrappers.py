# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple, Union, Any
import chex
import jax.numpy as jnp
from typing_extensions import NamedTuple, TypeAlias
from flax.struct import dataclass
from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.routing.robot_warehouse import RobotWarehouse
from jumanji.types import TimeStep
from jumanji.wrappers import Wrapper 

State: TypeAlias = Any

class Observation(NamedTuple):
    """The observation that the agent sees.
    agents_view: the agent's view of the environment.
    action_mask: boolean array specifying, for each agent, which action is legal.
    step_count: the number of steps elapsed since the beginning of the episode.
    """

    agents_view: chex.Array  # (num_agents, num_obs_features)
    action_mask: chex.Array  # (num_agents, num_actions)
    step_count: chex.Array  # (num_agents, )

class ObservationGlobalState(NamedTuple):
    """The observation seen by agents in centralised systems.
    Extends `Observation` by adding a `global_state` attribute for centralised training.
    global_state: The global state of the environment, often a concatenation of agents' views.
    """

    agents_view: chex.Array  # (num_agents, num_obs_features)
    action_mask: chex.Array  # (num_agents, num_actions)
    global_state: chex.Array  # (num_agents, num_agents * num_obs_features)
    step_count: chex.Array  # (num_agents, )

@dataclass
class LogEnvState:
    """State of the `LogWrapper`."""

    env_state: State
    episode_returns: chex.Numeric
    episode_lengths: chex.Numeric
    # Information about the episode return and length for logging purposes.
    episode_return_info: chex.Numeric
    episode_length_info: chex.Numeric

class MultiAgentWrapper(Wrapper):
    def __init__(self, env: Environment):
        super().__init__(env)
        self._num_agents = self._env.num_agents
        self.time_limit = self._env.time_limit

    def modify_timestep(self, timestep: TimeStep) -> TimeStep[Observation]:
        """Modify the timestep for `step` and `reset`."""
        pass

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep]:
        """Reset the environment."""
        state, timestep = self._env.reset(key)
        return state, self.modify_timestep(timestep)

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep]:
        """Step the environment."""
        state, timestep = self._env.step(state, action)
        return state, self.modify_timestep(timestep)

    def observation_spec(self) -> specs.Spec[Observation]:
        """Specification of the observation of the environment."""
        step_count = specs.BoundedArray(
            (self._num_agents,),
            jnp.int32,
            [0] * self._num_agents,
            [self.time_limit] * self._num_agents,
            "step_count",
        )
        return self._env.observation_spec().replace(step_count=step_count)

class RwareWrapper(MultiAgentWrapper):
    """Multi-agent wrapper for the Robotic Warehouse environment."""

    def __init__(self, env: RobotWarehouse):
        super().__init__(env)

    def modify_timestep(self, timestep: TimeStep) -> TimeStep[Observation]:
        """Modify the timestep for the Robotic Warehouse environment."""
        observation = Observation(
            agents_view=timestep.observation.agents_view,
            action_mask=timestep.observation.action_mask,
            step_count=jnp.repeat(timestep.observation.step_count, self._num_agents),
        )
        reward = jnp.repeat(timestep.reward, self._num_agents)
        discount = jnp.repeat(timestep.discount, self._num_agents)
        return timestep.replace(observation=observation, reward=reward, discount=discount)
    
class LogWrapper(Wrapper):
    """Log the episode returns and lengths."""

    def reset(self, key: chex.PRNGKey) -> Tuple[LogEnvState, TimeStep]:
        """Reset the environment."""
        state, timestep = self._env.reset(key)
        state = LogEnvState(state, jnp.float32(0.0), 0, jnp.float32(0.0), 0)
        return state, timestep

    def step(
        self,
        state: LogEnvState,
        action: chex.Array,
    ) -> Tuple[LogEnvState, TimeStep]:
        """Step the environment."""
        env_state, timestep = self._env.step(state.env_state, action)

        done = timestep.last()
        not_done = 1 - done

        new_episode_return = state.episode_returns + jnp.mean(timestep.reward)
        new_episode_length = state.episode_lengths + 1
        episode_return_info = state.episode_return_info * not_done + new_episode_return * done
        episode_length_info = state.episode_length_info * not_done + new_episode_length * done

        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * not_done,
            episode_lengths=new_episode_length * not_done,
            episode_return_info=episode_return_info,
            episode_length_info=episode_length_info,
        )
        return state, timestep


class AgentIDWrapper(Wrapper):
    """Add onehot agent IDs to observation."""

    def __init__(self, env: Environment, has_global_state: bool = False):
        super().__init__(env)
        self.has_global_state = has_global_state

    def _add_agent_ids(
        self, timestep: TimeStep, num_agents: int
    ) -> Union[Observation, ObservationGlobalState]:
        agent_ids = jnp.eye(num_agents)
        new_agents_view = jnp.concatenate([agent_ids, timestep.observation.agents_view], axis=-1)

        if self.has_global_state:
            # Add the agent IDs to the global state
            new_global_state = jnp.concatenate(
                [agent_ids, timestep.observation.global_state], axis=-1
            )

            return ObservationGlobalState(
                agents_view=new_agents_view,
                action_mask=timestep.observation.action_mask,
                step_count=timestep.observation.step_count,
                global_state=new_global_state,
            )

        else:
            return Observation(
                agents_view=new_agents_view,
                action_mask=timestep.observation.action_mask,
                step_count=timestep.observation.step_count,
            )

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep]:
        """Reset the environment."""
        state, timestep = self._env.reset(key)
        timestep.observation = self._add_agent_ids(timestep, self._env.num_agents)

        return state, timestep

    def step(
        self,
        state: State,
        action: chex.Array,
    ) -> Tuple[State, TimeStep]:
        """Step the environment."""
        state, timestep = self._env.step(state, action)
        timestep.observation = self._add_agent_ids(timestep, self._env.num_agents)

        return state, timestep

    def observation_spec(
        self,
    ) -> Union[specs.Spec[Observation], specs.Spec[ObservationGlobalState]]:
        """Specification of the observation of the `RobotWarehouse` environment."""
        obs_spec = self._env.observation_spec()
        num_obs_features = obs_spec.agents_view.shape[-1] + self._env.num_agents

        agents_view = specs.Array(
            (self._env.num_agents, num_obs_features), jnp.int32, "agents_view"
        )
        global_state = specs.Array(
            (
                self._env.num_agents,
                num_obs_features * self._env.num_agents + self._env.num_agents,
            ),
            jnp.int32,
            "global_state",
        )

        if self.has_global_state:
            return obs_spec.replace(agents_view=agents_view, global_state=global_state)
        return obs_spec.replace(agents_view=agents_view)


class GlobalStateWrapper(Wrapper):
    """Wrapper for adding global state to an environment that follows the mava API.

    The wrapper includes a global environment state to be used by the centralised critic.
    Note here that since most environments do not have a global state, we create one
    by concatenating the observations of all agents.
    """

    def modify_timestep(self, timestep: TimeStep) -> TimeStep[ObservationGlobalState]:
        global_state = jnp.concatenate(timestep.observation.agents_view, axis=0)
        global_state = jnp.tile(global_state, (self._env.num_agents, 1))

        observation = ObservationGlobalState(
            global_state=global_state,
            agents_view=timestep.observation.agents_view,
            action_mask=timestep.observation.action_mask,
            step_count=timestep.observation.step_count,
        )

        return timestep.replace(observation=observation)

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep]:
        """Reset the environment. Updates the step count."""
        state, timestep = self._env.reset(key)
        return state, self.modify_timestep(timestep)

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep]:
        """Step the environment. Updates the step count."""
        state, timestep = self._env.step(state, action)
        return state, self.modify_timestep(timestep)

    def observation_spec(self) -> specs.Spec[ObservationGlobalState]:
        """Specification of the observation of the `RobotWarehouse` environment."""

        obs_spec = self._env.observation_spec()
        num_obs_features = obs_spec.agents_view.shape[-1]
        global_state = specs.Array(
            (self._env.num_agents, self._env.num_agents * num_obs_features),
            jnp.int32,
            "global_state",
        )

        return specs.Spec(
            ObservationGlobalState,
            "ObservationSpec",
            agents_view=obs_spec.agents_view,
            action_mask=obs_spec.action_mask,
            global_state=global_state,
            step_count=obs_spec.step_count,
        )