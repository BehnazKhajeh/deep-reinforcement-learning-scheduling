import random
import numpy as np
import torch
from collections import deque


class ReplayBuffer:

    def __init__(self, capacity=100000):

        self.buffer = deque(
            maxlen=capacity
        )

    # =====================================
    # STORE EXPERIENCE
    # =====================================

    def push(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        next_valid_actions=None
    ):

        self.buffer.append(

            (
                state,
                action,
                reward,
                next_state,
                done,
                next_valid_actions
            )
        )

    # =====================================
    # SAMPLE MINI BATCH
    # =====================================

    def sample(self, batch_size):

        batch = random.sample(
            self.buffer,
            batch_size
        )

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        next_valid_actions = []

        for s, a, r, ns, d, nva in batch:

            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)
            next_valid_actions.append(nva)

        return (

            torch.FloatTensor(
                np.array(states)
            ),

            torch.LongTensor(
                np.array(actions)
            ),

            torch.FloatTensor(
                np.array(rewards)
            ),

            torch.FloatTensor(
                np.array(next_states)
            ),

            torch.FloatTensor(
                np.array(dones)
            ),

            next_valid_actions
        )
    # =====================================
    # BUFFER SIZE
    # =====================================

    def __len__(self):

        return len(self.buffer)
