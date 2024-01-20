# Copyright 2023 InstaDeep Ltd. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Any, Dict, Optional, Tuple
from collections import namedtuple
import numpy as np
import tensorflow as tf
import tree
from gymnasium.spaces import Box, Discrete

Sample = namedtuple('Sample', ['observations', 'actions', 'rewards', 'done', 'episode_return', 'legal_actions', 'env_state', 'zero_padding_mask'])

def get_schema(environment):
    act_type = list(environment.action_spaces.values())[0].dtype

    if isinstance(environment.action_spaces.values()[0], Discrete):
        act_shape = ()
        legals_shape  = environment.action_spaces.values()[0].n
    else:
        act_shape = list(environment.action_spaces.values())[0].shape
        legals_shape = act_shape

    obs_shape = list(environment.observation_spaces.values())[0].shape

    if "state" in environment.info_spec:
        state_shape = environment.info_spec["state"].shape
    else:
        state_shape = None

    schema = {}
    for agent in environment.possible_agents:
        schema[agent + "_observations"] = np.ones(obs_shape, np.float32)
        schema[agent + "_legal_actions"] = np.ones(legals_shape, np.float32)
        schema[agent + "_actions"] = np.ones(act_shape, act_type)
        schema[agent + "_rewards"] = np.array(1., np.float32)
        schema[agent + "_discounts"] = np.array(1., np.float32)

    ## Extras
    # Zero-padding mask
    schema["zero_padding_mask"] = np.array(1., np.float32)

    # Env state
    if state_shape is not None:
        schema["env_state"] = np.ones(state_shape, np.float32)

    # Episode return
    schema["episode_return"] = np.array(1., np.float32)

    return schema

class WriteSequence:
    def __init__(self, schema, sequence_length):
        self.schema = schema
        self.sequence_length = sequence_length
        self.numpy = tree.map_structure(
            lambda x: np.zeros(dtype=x.dtype, shape=(sequence_length, *x.shape)),
            schema,
        )
        self.t = 0

    def insert(self, agents, observations, legals, actions, rewards, discounts, env_state=None):
        assert self.t < self.sequence_length
        for agent in agents:
            self.numpy[agent + "_observations"][self.t] = observations[agent]

            self.numpy[agent + "_legal_actions"][self.t] = legals[agent]

            self.numpy[agent + "_actions"][self.t] = actions[agent]

            self.numpy[agent + "_rewards"][self.t] = rewards[agent]

            self.numpy[agent + "_discounts"][self.t] = discounts[agent]

        ## Extras
        # Zero padding mask
        self.numpy["zero_padding_mask"][self.t] = np.array(1, dtype=np.float32)

        # Global env state
        if "env_state" is not None:
            self.numpy["env_state"][self.t] = env_state

        # increment t
        self.t += 1

    def zero_pad(self, agents, episode_return):
        # Maybe zero pad sequence
        while self.t < self.sequence_length:
            for agent in agents:
                for item in [
                    "_observations",
                    "_legal_actions",
                    "_actions",
                    "_rewards",
                    "_discounts",
                ]:
                    self.numpy[agent + item][self.t] = np.zeros_like(
                        self.numpy[agent + item][0]
                    )

                ## Extras
                # Zero-padding mask
                self.numpy["zero_padding_mask"][self.t] = np.zeros_like(
                    self.numpy["zero_padding_mask"][0]
                )

                # Global env state
                if "env_state" in self.numpy:
                    self.numpy["env_state"][self.t] = np.zeros_like(
                        self.numpy["env_state"][0]
                    )

            # Increment time
            self.t += 1

        self.numpy["episode_return"] = np.array(episode_return, dtype="float32")


class MAOfflineEnvironmentSequenceLogger:
    def __init__(
        self,
        environment,
        sequence_length: int,
        period: int,
        logdir: str = "./offline_env_logs",
        label: str = "",
        min_sequences_per_file: int = 100000,
    ):
        self._environment = environment
        self._schema = get_schema(self._environment)

        self._active_buffer = []
        self._write_buffer = []

        self._min_sequences_per_file = min_sequences_per_file
        self._sequence_length = sequence_length
        self._period = period

        self._logdir = logdir
        self._label = label
        os.makedirs(logdir, exist_ok=True)

        self._observations = None
        self._legals = None
        self._env_state = None
        self._episode_return = None

        self._num_writes = 0
        self._timestep_ctr = 0

    def reset(self):
        """Resets the env and log the first timestep.

        Returns:
            dm.env timestep, extras
        """
        observations, info = self._environment.reset()

        if "env_state" in info:
            self._env_state  = info["env_state"]

        self._observations = observations

        self._episode_return = 0.0
        self._active_buffer = []
        self._timestep_ctr = 0

        return self._timestep, self._extras

    def step(self, actions):
        """Steps the env and logs timestep.

        Args:
            actions (Dict[str, np.ndarray]): actions per agent.

        Returns:
            dm.env timestep, extras
        """

        next_observations, rewards, terminals, truncations, info = self._environment.step(actions)

        if "env_state" in info:
            self._env_state  = info["env_state"]

        self._episode_return += np.mean(list(rewards.values()))

        # Log timestep
        self._log_timestep(
            self._observations, self._extras, next_timestep, actions, self._episode_return
        )
        self._timestep = next_timestep
        self._extras = next_extras

        return self._timestep, self._extras

    def _log_timestep(
        self,
        observations,
        legals,
        actions,
        rewards,
        env_state,
        episode_return: float,
    ) -> None:
        if self._timestep_ctr % self._period == 0:
            self._active_buffer.append(
                WriteSequence(
                    schema=self._schema, sequence_length=self._sequence_length
                )
            )

        for write_sequence in self._active_buffer:
            if write_sequence.t < self._sequence_length:
                write_sequence.insert(
                    self._agents, observations, legals, actions, rewards, discounts
                )

        if next_timestep.last():
            for write_sequence in self._active_buffer:
                write_sequence.zero_pad(self._agents, episode_return)
                self._write_buffer.append(write_sequence)
        if len(self._write_buffer) >= self._min_sequences_per_file:
            self._write()

        # Increment timestep counter
        self._timestep_ctr += 1

    def _write(self) -> None:
        filename = os.path.join(
            self._logdir, f"{self._label}_sequence_log_{self._num_writes}.tfrecord"
        )
        with tf.io.TFRecordWriter(filename, "GZIP") as file_writer:
            for write_sequence in self._write_buffer:

                # Convert numpy to tf.train features
                dict_of_features = tree.map_structure(
                    self._numpy_to_feature, write_sequence.numpy
                )

                # Create Example for writing
                features_for_example = tf.train.Features(feature=dict_of_features)
                example = tf.train.Example(features=features_for_example)

                # Write to file
                file_writer.write(example.SerializeToString())

        # Increment write counter
        self._num_writes += 1

        # Flush buffer and reset ctr
        self._write_buffer = []

    def _numpy_to_feature(self, np_array: np.ndarray):
        tensor = tf.convert_to_tensor(np_array)
        serialized_tensor = tf.io.serialize_tensor(tensor)
        bytes_list = tf.train.BytesList(value=[serialized_tensor.numpy()])
        feature_of_bytes = tf.train.Feature(bytes_list=bytes_list)
        return feature_of_bytes

    def __getattr__(self, name: str) -> Any:
        """Expose any other attributes of the underlying environment.

        Args:
            name (str): attribute.

        Returns:
            Any: return attribute from env or underlying env.
        """
        if hasattr(self.__class__, name):
            return self.__getattribute__(name)
        else:
            return getattr(self._environment, name)
        

class OfflineLogger:

    def __init__(self, env):

        self.environment = MAOfflineEnvironmentSequenceLogger(env, 20, 10, min_sequences_per_file=2)

    def __getattr__(self, name: str) -> Any:
        """Expose any other attributes of the underlying environment.

        Args:
            name (str): attribute.

        Returns:
            Any: return attribute from env or underlying env.
        """
        if hasattr(self.__class__, name):
            return self.__getattribute__(name)
        else:
            return getattr(self.environment, name)
