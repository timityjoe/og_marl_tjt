# Copyright 2023 InstaDeep Ltd. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

import time

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Sequence

import flashbax as fbx
import jax
import jax.numpy as jnp
import jax.random as jrand
import jaxmarl
import jumanji
import optax
import tree
import wandb
from flashbax.buffers.trajectory_buffer import TrajectoryBufferState
from flashbax.vault import Vault
from flax import linen as nn
from jaxmarl.environments.smax import map_name_to_scenario
from jumanji.environments.routing.robot_warehouse.generator import RandomGenerator

from og_marl.environments.utils import get_environment
from og_marl.jax.jax_marl_wrapper import JaxMarlState, JaxMarlWrapper
from og_marl.jax.jumanji_wrappers import AgentIDWrapper, LogWrapper, RwareWrapper
from og_marl.jax.tf_dataset_to_flashbax import FlashbaxBufferStore


def train_bc_system(
    logger,
    seed: int = 42,
    learning_rate: float = 3e-4,
    batch_size: float = 32,
    policy_layer_sizes: Sequence[int] = (64,),
    policy_gru_layer_size: int = 64,
    num_epochs: int = 1000,
    num_training_steps_per_epoch: int = 1000,
    num_episodes_per_evaluation: int = 4,
    json_writer=None
):

    ##################
    ##### Config #####
    ##################

    BATCH_SIZE = batch_size
    SEQUENCE_LENGTH = 2
    LR = learning_rate
    LAYER_SIZES = policy_layer_sizes
    GRU_LAYER_SIZE = policy_gru_layer_size
    NUM_TRAIN_STEPS_PER_EPOCH = num_training_steps_per_epoch
    NUM_EVALS = num_episodes_per_evaluation
    SEED = seed
    NUM_EPOCHS = num_epochs

    NUM_ACTS = 10
    NUM_AGENTS = 5

    ##################
    ### End Config ###
    ##################

    class BehaviourCloningPolicy(nn.Module):
        dense_layer_sizes: Sequence[int]
        gru_hidden_size: int
        output_size: int

        @nn.compact
        def __call__(self, carry, inputs, done):
            dense_layers = [nn.Dense(size) for size in self.dense_layer_sizes]
            x = inputs
            for layer in dense_layers:
                x = layer(x)
                x = nn.relu(x)
            # carry, x = nn.GRUCell(self.gru_hidden_size)(carry, x)

            # Maybe reinitialise carry
            carry = jnp.where(
                jnp.expand_dims(done, axis=-1),
                self.initialize_carry(self.gru_hidden_size, inputs.shape),
                carry,
            )

            # Final dense layer for output
            output = nn.Dense(self.output_size)(x)
            
            return carry, output

        @staticmethod
        def initialize_carry(layer_size, input_shape):
            """Initializes the carry state."""
            # Use a dummy key since the default state init fn is just zeros.
            return nn.GRUCell(layer_size).initialize_carry(jax.random.PRNGKey(0), input_shape)

    def softmax_cross_entropy_loss(index, logits): # softmax cross entropy loss
        num_labels = logits.shape[-1]
        labels = nn.one_hot(index, num_labels)
        probs = nn.softmax(logits)
        return -jnp.sum(labels * jnp.log(probs + 1e-12), axis=-1) # small constant for numerical stability

    def unroll_policy(params, obs_seq, done_seq):
        f = lambda carry, inputs: BehaviourCloningPolicy(LAYER_SIZES, GRU_LAYER_SIZE, NUM_ACTS).apply(params, carry, inputs[0], inputs[1])
        init_carry = BehaviourCloningPolicy(LAYER_SIZES, GRU_LAYER_SIZE, NUM_ACTS).initialize_carry(GRU_LAYER_SIZE, obs_seq.shape[-1:])
        carry, logits = jax.lax.scan(f, init_carry, (obs_seq, done_seq))
        return logits

    def behaviour_cloning_loss(params, obs_seq, act_seq, done_seq, mask):
        logits = unroll_policy(params, obs_seq, done_seq)
        logits = jnp.where(
            jnp.expand_dims(mask, axis=-1),
            logits,
            jnp.ones_like(logits),
        )  # avoid nans, get masked out later
        loss = jax.vmap(softmax_cross_entropy_loss)(act_seq, logits)
        return jnp.sum(loss * mask) / jnp.sum(mask) # masked mean

    def batched_multi_agent_behaviour_cloninig_loss(params, obs_seq, act_seq, done_seq, mask):
        """
        Args:
            params: a container of params for the behaviour cloning network which is shared between 
                all agents in the sustem.
            obs_seq: an array of a sequence of observations for all agents. Shape (B,N,T,O) where
                B is the batch dim, N is the number of agents, T is the time dimension and O is
                the observation dim.
            act_seq: is an array of a sequence of actions for all agents. Shape (B,N,T).
        Returns: 
            A scalar behaviour cloning loss.
        """
        multi_agent_behaviour_cloning_loss = jax.vmap(behaviour_cloning_loss, (None,1,1,1,None)) # vmap over agent dim which is after time dim
        batched_multi_agent_behaviour_cloninig_loss = jax.vmap(multi_agent_behaviour_cloning_loss, (None, 0,0,0,0))
        loss = batched_multi_agent_behaviour_cloninig_loss(params, obs_seq, act_seq, done_seq, mask)
        return jnp.mean(loss)

    def train_epoch(rng_key, params, opt_state, buffer_state):
        buffer = fbx.make_trajectory_buffer(
            # Unused when reload:
            add_batch_size=1,
            max_length_time_axis=SEQUENCE_LENGTH,
            min_length_time_axis=SEQUENCE_LENGTH,
            # Important:
            period=1, # or 1?
            sample_batch_size=BATCH_SIZE,
            sample_sequence_length=SEQUENCE_LENGTH,
        )
        optim = optax.chain(optax.clip_by_global_norm(10), optax.adam(LR))

        def train_step(carry, rng_key):
            params, opt_state, buffer_state = carry
            batch = buffer.sample(buffer_state, rng_key)
            loss, grads = jax.value_and_grad(batched_multi_agent_behaviour_cloninig_loss)(params, batch.experience["obs"], batch.experience["act"], batch.experience["done"], batch.experience["mask"])
            updates, opt_state = optim.update(
                grads, opt_state, params
            )
            params = optax.apply_updates(params, updates)
            return (params, opt_state, buffer_state), loss

        init_carry = (params, opt_state, buffer_state)
        rng_keys = jax.random.split(rng_key, num=NUM_TRAIN_STEPS_PER_EPOCH)
        carry, loss = jax.lax.scan(train_step, init_carry, rng_keys)
        params, opt_state, buffer_state = carry
        return params, opt_state, {"loss": loss}

    @jax.jit
    def select_actions(carry, params, obs, legals, done):
        policy = BehaviourCloningPolicy(LAYER_SIZES, GRU_LAYER_SIZE, NUM_ACTS)
        carry, logits = policy.apply(params, carry, obs, done)
        logits = jnp.where(legals, logits, -99999999.) # Legal action masking
        act = jnp.argmax(logits, axis=-1)
        return carry, act

    def evaluation(rng_key, init_carry, params, env_reset_fn, env_step_fn):
        episode_returns = []
        for e in range(NUM_EVALS):
            rng_key, eval_key = jax.random.split(rng_key, 2)
            env_state, timestep = env_reset_fn(eval_key)
            episode_return = 0
            done = jnp.array(False)
            carry = init_carry
            while not done:
                obs = timestep.observation.agents_view
                legals = timestep.observation.action_mask
                carry, act = select_actions(carry, params, obs, legals, done)

                env_state, timestep = env_step_fn(env_state, act)

                done = timestep.last()
                episode_return += jnp.mean(timestep.reward)
            episode_returns.append(episode_return)
        return {"evaluator/episode_return": sum(episode_returns)/NUM_EVALS}
    
    def transform_buffer_state(state):
        experience = {}
        experience["obs"] = state.experience["observation"]
        experience["act"] = state.experience["action"]
        experience["rew"] = state.experience["reward"]
        experience["done"] = state.experience["done"]
        experience["legals"] = state.experience["legal_action_mask"]
        experience["mask"] = jnp.ones((*experience["act"].shape[:2],))
        experience["env_state"] = jnp.reshape(state.experience["observation"], (*state.experience["observation"].shape[:2],-1))
        state = TrajectoryBufferState(experience=experience, is_full=state.is_full, current_index=state.current_index)
        return state
    
    # Create envs
    def make_env():
        # task_config = {"column_height": 8, "shelf_rows": 1, "shelf_columns": 3, "num_agents": 2, "sensor_range": 1, "request_queue_size": 2}
        # generator = RandomGenerator(**task_config)
        # env = jumanji.make("RobotWarehouse-v0", generator=generator)
        # env = RwareWrapper(env)
        # env = AgentIDWrapper(env)
        # env = LogWrapper(env)

        # Placeholder for creating JAXMARL environment.
        env = JaxMarlWrapper(jaxmarl.make("HeuristicEnemySMAX", scenario=map_name_to_scenario("2s3z")))
        env = AgentIDWrapper(env)
        # env = GlobalStateWrapper(env)
        env = LogWrapper(env)

        return env

    ################
    ##### MAIN #####
    ################

    wandb.init(project="jax-og-marl")
    rng_key = jax.random.PRNGKey(SEED)

    vault = Vault(vault_name="ff_ippo", vault_uid="20231208111927")
    buffer_state = vault.read(percentiles=(50, 100))
    buffer_state = transform_buffer_state(buffer_state)

    dummy_obs = buffer_state.experience["obs"][0,0,0]
    dummy_done = jnp.array([False,False])
    policy = BehaviourCloningPolicy(LAYER_SIZES, GRU_LAYER_SIZE, NUM_ACTS)
    init_carry = policy.initialize_carry(GRU_LAYER_SIZE, dummy_obs.shape)
    params = policy.init(rng_key, init_carry, dummy_obs, dummy_done)
    opt_state = optax.chain(optax.clip_by_global_norm(10), optax.adam(3e-4)).init(params)

    # Make the environment for evaluation
    env = make_env()
    env_step_fn = jax.jit(env.step)
    env_reset_fn = jax.jit(env.reset)

    for i in range(NUM_EPOCHS):
        rng_key, train_key, eval_key = jax.random.split(rng_key, 3)
        eval_logs = evaluation(eval_key, init_carry, params, env_reset_fn, env_step_fn)
        logger.write(eval_logs, force=True)
        if json_writer is not None:
            json_writer.write(
                (i+1) * NUM_TRAIN_STEPS_PER_EPOCH,
                "evaluator/episode_return",
                eval_logs["evaluator/episode_return"],
                i
            )

        start_time = time.time()
        rng_key, train_key = jax.random.split(rng_key)
        params, opt_state, logs = train_epoch(train_key, params, opt_state, buffer_state)
        end_time = time.time()

        logs["loss"] = jnp.mean(logs["loss"])
        logs["Trainer Steps"] = (i+1) * NUM_TRAIN_STEPS_PER_EPOCH
        if i != 0: # don't log SPC when tracing
            logs["Train SPS"] = 1 / ((end_time - start_time) / NUM_TRAIN_STEPS_PER_EPOCH)

        logger.write(logs)

    eval_logs = evaluation(eval_key, init_carry, params, env_reset_fn, env_step_fn)
    logger.write(eval_logs, force=True)
    if json_writer is not None:
        eval_logs = {f"absolute/{key.split('/')[1]}": value for key, value in eval_logs.items()}
        json_writer.write(
            (i+1) * NUM_TRAIN_STEPS_PER_EPOCH,
            "absolute/episode_return",
            eval_logs["absolute/episode_return"],
            i
        )

    print("Done")