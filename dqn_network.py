import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingDQN(nn.Module):

    def __init__(self, state_dim, action_dim):

        super().__init__()

        # =========================
        # shared feature layer
        # =========================

        self.fc1 = nn.Linear(
            state_dim,
            128
        )

        self.fc2 = nn.Linear(
            128,
            128
        )

        # =========================
        # value stream
        # =========================

        self.value_fc = nn.Linear(
            128,
            64
        )

        self.value = nn.Linear(
            64,
            1
        )

        # =========================
        # advantage stream
        # =========================

        self.adv_fc = nn.Linear(
            128,
            64
        )

        self.advantage = nn.Linear(
            64,
            action_dim
        )

    def forward(self, x):

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        # =========================
        # value stream
        # =========================

        v = F.relu(
            self.value_fc(x)
        )

        v = self.value(v)

        # =========================
        # advantage stream
        # =========================

        a = F.relu(
            self.adv_fc(x)
        )

        a = self.advantage(a)

        # =========================
        # combine
        # =========================

        q = v + (
            a - a.mean(
                dim=1,
                keepdim=True
            )
        )

        return q
