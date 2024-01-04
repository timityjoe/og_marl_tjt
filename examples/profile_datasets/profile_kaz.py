from og_marl.environments import kaz
from og_marl.utils.dataset_utils import profile_dataset

scenario = "kaz" 

env = kaz.KAZ()

dataset = env.get_dataset()

stats = profile_dataset(dataset)

print("\DATASET STATS")
print(stats)

print("\DATASET SAMPLE")
dataset = iter(dataset)
sample = next(dataset)

print()
print("!!! Note that samples are sequences of consecutive timesteps. So the leading dimension is the time dimension !!!")
print()

agent = env.agents[0]
print(f"Agent_0 Observation Shape: {sample.observations[agent].observation.shape}")
print(f"Agent_0 Action Shape: {sample.actions[agent].shape}")
print(f"Agent_0 Reward Shape: {sample.rewards[agent].shape}")
print(f"Agent_0 Reward Shape: {sample.rewards[agent].shape}")
print(f"Agent_0 Discount Shape: {sample.discounts[agent].shape}")