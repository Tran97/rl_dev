# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
import gymnasium
import numpy as np
from gymnasium.spaces.discrete import Discrete
from irlc.ex01.agent import Agent, train
import random

class BobFriendEnvironment(gymnasium.Env): 
    def __init__(self, x0=20):
        self.x0 = x0
        self.action_space = Discrete(2)     # Possible actions {0, 1} 
        self.probability_u1 = 3/4
        self.probability_u0 = 1/4

    def reset(self):
        # TODO: 1 lines missing.
        #raise NotImplementedError("Insert your solution and remove this error.")
        self.s = 0
        return self.s, {}

    def step(self, a: int):
        # TODO: 9 lines missing.
        if a == 0:
            s_next: float = self.x0 * 1.1
            reward: float = s_next - self.x0
            terminated = True
        else:
            # Randomly choose between u=1 and u=0 based on probabilities
            chosen_action = random.choices([1, 0], weights=[self.probability_u1, self.probability_u0])[0]
            # Print the chosen action
            if chosen_action == 1:
                s_next: float = self.x0 + 12
                reward: float = s_next - self.x0
                terminated = True
            else:
                s_next: float = 0
                reward: float = s_next - self.x0
                terminated = True
        #raise NotImplementedError("Insert your solution and remove this error.")
        self.s = s_next
        return s_next, reward, terminated, False, {}

class AlwaysAction_u0(Agent):
    def pi(self, s, k, info=None):  
        """This agent should always take action u=0."""
        ## TODO: 1 lines missing.
        #raise NotImplementedError("Implement function body")
        return 0
class AlwaysAction_u1(Agent):
    def pi(self, s, k, info=None):  
        """This agent should always take action u=1."""
        # TODO: 1 lines missing.
        #raise NotImplementedError("Implement function body")
        return 1

def run_bob():
    # Part A:
    env = BobFriendEnvironment()
    x0, _ = env.reset()
    print(f"Initial amount of money is x0 = {x0} (should be 20 kroner)")
    print("Lets put it in the bank, we should end up in state x1=22 and get a reward of 2 kroner")
    x1, reward, _, _, _ = env.step(0)
    print("we got", x1, reward)
    # Since we reset the environment, we should get the same result as before:
    env.reset()
    x1, reward, _, _, _ = env.step(0)
    print("(once more) we got", x1, reward, "(should be the same as before)")

    env.reset()  # We must call reset -- the environment has possibly been changed!
    print("Lets lend it to our friend -- what happens will now be random")
    x1, reward, _, _, _ = env.step(1)
    print("we got", x1, reward)

    # Part B:
    stats, _ = train(env, AlwaysAction_u0(env), num_episodes=1000)
    average_u0 = np.mean([stat['Accumulated Reward'] for stat in stats])

    stats, _ = train(env, AlwaysAction_u1(env), num_episodes=1000)
    average_u1 = np.mean([stat['Accumulated Reward'] for stat in stats])
    print(f"Average reward while taking action u=0 was {average_u0} (should be 2)")
    print(f"Average reward while taking action u=1 was {average_u1} (should be 12)")


if __name__ == "__main__":
    run_bob()
