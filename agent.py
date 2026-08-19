
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from dqn_network import DuelingDQN


class Agent:

    def __init__(

            self,
            state_dim,
            action_dim,
            lr=0.001,
            gamma=0.99,
            batch_size=64,
            use_action_masking=True
    ):
        self.global_step = 0
        self.target_update_freq = 1000  


        self.device = torch.device(

            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        self.use_action_masking = (
            use_action_masking
        )

        # =====================================
        # MAIN NETWORK
        # =====================================

        self.q_net = DuelingDQN(
            state_dim,
            action_dim
        ).to(self.device)

        print("=" * 60)
        print("GPU MEMORY AFTER MODEL LOADING")
        print(torch.cuda.memory_summary())
        print("=" * 60)

        # =====================================
        # TARGET NETWORK
        # =====================================

        self.target_net = DuelingDQN(
            state_dim,
            action_dim
        ).to(self.device)

        self.target_net.load_state_dict(
            self.q_net.state_dict()
        )

        # =====================================
        # OPTIMIZER
        # =====================================

        self.optimizer = torch.optim.Adam(

            self.q_net.parameters(),
            lr=lr
        )

        # =====================================
        # RL PARAMETERS
        # =====================================
        self.gamma = gamma
        # self.gamma = 0.99
        self.batch_size = batch_size

        self.epsilon = 1.0

        self.epsilon_min = 0.05

        self.epsilon_decay = 0.995

        self.action_dim = action_dim


    def select_action(
            self,
            state,
            valid_actions
    ):

        if random.random() < self.epsilon:
            return random.choice(
                valid_actions
            )

        state = torch.FloatTensor(
            state
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.q_net(state)

        if self.use_action_masking:

            action = max(
                valid_actions,
                key=lambda a: q_values[0][a].item()
            )

        else:

            action = torch.argmax(
                q_values
            ).item()

        return action
    # =================================================
    # TRAIN
    # =================================================

    def train(
        self,
        replay_buffer,
        batch_size=64
    ):

        if len(replay_buffer) < self.batch_size:
            return

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            next_valid_actions

        ) = replay_buffer.sample(self.batch_size)

        # =====================================
        # TO DEVICE
        # =====================================

        states = states.to(self.device)

        actions = actions.to(self.device)

        rewards = rewards.to(self.device)

        next_states = next_states.to(self.device)

        dones = dones.to(self.device)

        # =====================================
        # CURRENT Q
        # =====================================

        current_q = self.q_net(states)

        current_q = current_q.gather(
            1,
            actions.unsqueeze(1)
        ).squeeze(1)

        # =====================================
        # DOUBLE DQN
        # =====================================

        with torch.no_grad():

            next_q_online = self.q_net(
                next_states
            )

            if self.use_action_masking:

                mask = torch.full_like(
                    next_q_online,
                    -1e9
                )

                for i, valid_actions in enumerate(
                    next_valid_actions
                ):

                    if valid_actions is None:
                        mask[i, :] = 0

                    else:

                        for action in valid_actions:
                            mask[i, action] = 0

                next_q_online = (
                    next_q_online
                    + mask
                )

            # action selection
            next_actions = next_q_online.argmax(1)

            # action evaluation
            next_q = self.target_net(
                next_states
            ).gather(

                1,

                next_actions.unsqueeze(1)

            ).squeeze(1)

            target_q = rewards + (

                self.gamma
                * next_q
                * (1 - dones)
            )

        # =====================================
        # LOSS
        # =====================================
        if self.global_step == 0:
            print("=" * 60)
            print("GPU MEMORY BEFORE FIRST BACKWARD")
            print(torch.cuda.memory_summary())
            print("=" * 60)

        loss = F.mse_loss(
            current_q,
            target_q
        )

        # =====================================
        # OPTIMIZATION
        # =====================================

        self.optimizer.zero_grad()

        loss.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.q_net.parameters(),
            1.0
        )

        self.optimizer.step()
        self.global_step += 1

        if self.global_step % self.target_update_freq == 0:
            self.update_target()
    # =================================================
    # TARGET UPDATE
    # =================================================

    def update_target(self):

        self.target_net.load_state_dict(
            self.q_net.state_dict()
        )

    # =================================================
    # EPSILON DECAY
    # =================================================

    def decay_epsilon(self):

        if self.epsilon > self.epsilon_min:

            self.epsilon *= (
                self.epsilon_decay
            )

    def select_greedy_action(
            self,
            state,
            valid_actions
    ):

        state = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            q = self.q_net(state).squeeze(0)

        mask = torch.full_like(
            q,
            -1e9
        )

        mask[valid_actions] = 0

        q = q + mask

        return q.argmax().item()
