import pygame
import time
import dm_env

from og_marl.offline_tools import OfflineLogger
from og_marl.environments.kaz import KAZ

class ManualPolicy:
    def __init__(self, env, show_obs: bool = False):
        self.env = env
        self.agents = self.env.agents

        # TO-DO: show current agent observation if this is True
        self.show_obs = show_obs

        # action mappings for all agents are the same
        self.action_mapping = [{}, {}]

        self.default_action = 5
        self.action_mapping[0][pygame.K_w] = 0  # front
        self.action_mapping[0][pygame.K_s] = 1  # back
        self.action_mapping[0][pygame.K_a] = 2  # rotate left
        self.action_mapping[0][pygame.K_d] = 3  # rotate right
        self.action_mapping[0][pygame.K_f] = 4  # weapon

        self.action_mapping[1][pygame.K_UP] = 0  # front
        self.action_mapping[1][pygame.K_DOWN] = 1  # back
        self.action_mapping[1][pygame.K_LEFT] = 2  # rotate left
        self.action_mapping[1][pygame.K_RIGHT] = 3  # rotate right
        self.action_mapping[1][pygame.K_r] = 4  # weapon

    def __call__(self):
        # only trigger when we are the correct agent

        # set the default action
        actions = [self.default_action, self.default_action]

        # if we get a key, override action using the dict
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key in self.action_mapping[0]:
                    actions[0] = self.action_mapping[0][event.key]
                elif event.key in self.action_mapping[1]:
                    actions[1] = self.action_mapping[1][event.key]

        actions = {self.agents[i]: actions[i] for i in range(2)}

        return actions

    @property
    def available_agents(self):
        return self.env.agent_name_mapping


if __name__ == "__main__":
    env = KAZ(render=True)
    env = OfflineLogger(env, sequences_per_file=20)

    timestep, _ = env.reset()

    # Instantiate controller that maps to keyboard for humans
    manual_policy = ManualPolicy(env)

    while True:

        # Get actions from keyboard
        actions = manual_policy()

        time.sleep(0.08) # slow down the game for humans

        timestep, _ = env.step(actions)

        if timestep.step_type == dm_env.StepType.LAST:
            env.reset()