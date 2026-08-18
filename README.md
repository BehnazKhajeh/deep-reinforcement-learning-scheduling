# Reinforcement Learning for Intelligent Workflow Scheduling

This repository contains an experimental implementation of a **Deep Reinforcement Learning (DRL) framework for workflow scheduling in cloud computing environments**.

The project was developed to investigate how reinforcement learning can be used to make sequential scheduling decisions while considering multiple system-level objectives such as execution time, resource utilization, cost, and security-related constraints.

> **Note:** This repository is intended to demonstrate the implementation and engineering aspects of the reinforcement-learning framework. Some research-specific details, experimental configurations, and methodological components are intentionally omitted because the associated research is currently under review.

---

## Overview

Workflow scheduling is a sequential decision-making problem in which computational tasks must be assigned to available virtual machines while respecting workflow dependencies and infrastructure constraints.

In this project, the scheduling problem is modeled as a **Markov Decision Process (MDP)**.

At each scheduling step, the reinforcement-learning agent observes the current state of the workflow and cloud infrastructure and selects an appropriate virtual machine for the next schedulable task.

The environment then:

1. identifies the current ready task,
2. determines feasible VM candidates,
3. evaluates resource availability,
4. assigns the task to the selected VM,
5. determines its execution interval,
6. updates the scheduling state,
7. calculates the reward,
8. and exposes the resulting state to the agent.

This process is repeated until all workflow tasks have been scheduled.

---

## Reinforcement Learning Framework

The main implementation is based on a **Dueling Deep Q-Network (Dueling DQN)** architecture implemented in **PyTorch**.

The agent learns a Q-value for each possible VM-selection action and gradually improves its scheduling policy through interaction with the simulated cloud environment.

The implementation includes several standard components of Deep Q-Learning:

* Dueling network architecture
* Experience replay
* Target network
* Epsilon-greedy exploration
* Discounted rewards
* Batched optimization
* Action masking
* Multiple training seeds
* Deterministic evaluation

The implementation is designed so that different scheduling variants can be evaluated under the same workflow and infrastructure configuration.

---

## Dueling DQN

The Dueling DQN architecture separates the estimation of:

* the **state value**, and
* the **relative advantage of individual actions**.

Conceptually, the network can be represented as:

```text
                    State
                      │
                      ▼
              Shared Feature Layers
                      │
             ┌────────┴────────┐
             ▼                 ▼
        Value Stream       Advantage Stream
             │                 │
             └────────┬────────┘
                      ▼
                 Q-Values
                      │
                      ▼
                VM Selection
```

This architecture allows the agent to learn both the quality of the current scheduling state and the relative usefulness of different VM-selection decisions.

---

## Constrained Action Selection

One of the important implementation aspects of the environment is **feasibility-aware action selection**.

Before an action is selected, candidate VMs can be filtered according to constraints such as:

* CPU availability
* memory availability
* security requirements
* risk-related admissibility

This prevents the agent from selecting actions that are known to be infeasible.

The general decision process is therefore:

```text
Current Workflow State
        │
        ▼
Generate VM Candidates
        │
        ▼
Check Resource Constraints
        │
        ▼
Apply Admissibility Rules
        │
        ▼
Valid VM Candidates
        │
        ▼
Dueling DQN
        │
        ▼
Selected VM
        │
        ▼
Environment Transition
```

This separation between **feasibility** and **preference learning** is useful for studying constrained reinforcement-learning approaches for scheduling.

---

## Environment

The scheduling environment represents a workflow as a directed acyclic graph (DAG).

Each task contains information related to:

* execution requirements
* CPU demand
* memory demand
* security sensitivity
* workflow dependencies

Virtual machines are represented using characteristics such as:

* CPU capacity
* memory capacity
* computational cost
* security-related attributes
* risk characteristics

The environment maintains resource calendars so that task placement is performed over feasible execution intervals rather than simply selecting a VM based on static capacity.

---

## State Representation

The state representation combines three types of information:

### Task-level information

The current task is represented using features describing its computational and workflow characteristics.

### VM-level candidate information

Each candidate VM is represented using features describing its current scheduling consequences, including temporal and resource-related information.

### Global scheduling information

The state also contains global information describing the current progress of the scheduling process and accumulated system metrics.

This **candidate-conditioned representation** allows the neural network to compare the immediate consequences of different VM-selection actions.

The exact research-specific state construction is intentionally not disclosed in this public repository.

---

## Reward Design

The reward function is designed as a multi-objective scheduling signal.

The implementation considers several factors, including:

* makespan
* computational cost
* risk-related exposure
* budget overflow

The general form can be viewed conceptually as:

```text
Reward
   │
   ├── Scheduling-time component
   ├── Cost component
   ├── Risk component
   └── Budget component
```

The relative importance of these components can be modified experimentally through configuration parameters.

This makes the implementation suitable for studying how different objective trade-offs affect the learned scheduling policy.

---

## Experimental Variants

The repository contains several implementations used to compare different aspects of the scheduling approach.

Examples include:

| Variant            | Purpose                                           |
| ------------------ | ------------------------------------------------- |
| `D3QN_Base`        | Baseline Deep RL scheduling                       |
| `D3QN_RiskPenalty` | RL with an explicit risk-related reward component |
| `D3QN_RiskMask`    | RL with risk-aware action filtering               |
| `Proposed`         | Combined constrained RL configuration             |

The exact research formulation and final experimental interpretation are intentionally kept outside this public README.

---

## Workflows

The implementation has been tested with workflow DAGs used in scientific workflow scheduling research.

Example workflow families include:

* Epigenomics
* Inspiral
* Montage
* SIPHT

The workflow descriptions are represented using **DAX files** and are parsed to construct the corresponding DAG and task/resource information.

---

## Implementation

The project is implemented primarily in:

* **Python**
* **PyTorch**
* NumPy
* Pandas
* Matplotlib

The codebase includes components for:

```text
Workflow Parsing
       │
       ▼
Scheduling Environment
       │
       ├── Task Management
       ├── VM Resource Calendars
       ├── Constraint Checking
       └── Reward Calculation
       │
       ▼
Replay Buffer
       │
       ▼
Dueling DQN Agent
       │
       ├── Q-Network
       ├── Target Network
       ├── Experience Replay
       └── Epsilon-Greedy Exploration
       │
       ▼
Training
       │
       ▼
Policy Evaluation
       │
       ▼
Metrics & Visualization
```

---

## Reproducible Training

The training pipeline supports multiple random seeds to reduce the effect of stochastic initialization and exploration.

A typical experiment can be configured using:

```python
SEEDS = [11, 22, 33, 44, 55]

for seed in SEEDS:
    train_model(
        risk_weight=0.1,
        train_seed=seed,
        episodes=500
    )
```

Training results are stored for subsequent analysis and visualization.

Multiple seeds are used to obtain more reliable estimates of the behavior of the learned policy rather than relying on a single training run.

---

## Evaluation

After training, the learned model is evaluated separately from the training process.

The evaluation stage uses deterministic policy execution with exploration disabled and can be performed using independent evaluation seeds.

The evaluation pipeline records metrics such as:

* Makespan
* Total cost
* Security compliance
* Security violations
* Risk exposure
* Budget violations
* Invalid actions
* Inference runtime

This separation between **training metrics** and **final evaluation metrics** is important because the performance of the final model should not be judged only from the episodes observed during learning.

---

## Results

The repository contains scripts for analyzing and visualizing the experimental results.

Typical analyses include:

* Makespan comparison
* Cost comparison
* Risk exposure
* Security compliance
* Multi-seed variability
* Risk-weight sensitivity
* Ablation studies
* Risk–performance trade-offs

Example output files include:

```text
results/
├── main_experiment/
├── Multi_seed/
├── ablation.csv
├── main_experiment_summary.csv
└── visualizations/
```

The numerical results and research-specific conclusions are intentionally not reproduced in this README.

---

## Why PyTorch?

PyTorch was selected because it provides a flexible framework for implementing and experimenting with Deep Q-Learning architectures.

In particular, the project makes use of:

* `torch.nn`
* custom neural-network modules
* tensor-based state representations
* GPU acceleration
* gradient-based optimization
* model checkpointing
* reproducible random seeds

The implementation therefore provides practical experience with building a complete DRL pipeline rather than only using a high-level reinforcement-learning library.

---

## Research Skills Demonstrated

This repository demonstrates practical experience with:

### Deep Reinforcement Learning

* DQN
* Double DQN concepts
* Dueling DQN
* Experience replay
* Target networks
* Epsilon-greedy exploration

### PyTorch

* Neural-network implementation
* Training loops
* Optimization
* Tensor operations
* GPU execution
* Model saving/loading

### Scheduling

* DAG-based workflow representation
* Resource-aware placement
* VM selection
* Resource calendars
* Multi-objective optimization

### Experimental Research

* Ablation studies
* Multiple random seeds
* Independent evaluation
* Metric aggregation
* Statistical summaries
* Visualization

---

## Repository Structure

A simplified structure of the project is:

```text
.
├── agent.py
├── train.py
├── environment/
│   └── workflow_env.py
├── workflows/
│   ├── Epigenomics.dax
│   ├── Inspiral.dax
│   ├── MONTAGE.dax
│   └── SIPHT.dax
├── experiments/
├── evaluation/
├── plotting/
├── results/
└── README.md
```

The exact internal research implementation may differ from this simplified public structure.

---

## Research Status

This repository is associated with ongoing research on **reinforcement-learning-based scheduling for cloud computing**.

The public version focuses on demonstrating:

> **Practical implementation of Deep Reinforcement Learning for constrained workflow scheduling using PyTorch.**

Research-specific methodological details, complete experimental configurations, and unpublished findings are intentionally limited in this public version.

---

## Disclaimer

This repository is primarily intended to demonstrate the implementation, experimentation, and engineering aspects of the project.

The code and experiments are provided for research and educational purposes. The public repository should not be interpreted as a complete disclosure of the associated research methodology or unpublished scientific contribution.
