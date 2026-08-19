import numpy as np
import torch
import pandas as pd
import os
import random
import json
from collections import Counter
from pathlib import Path

from env_workflow_env import WorkflowEnv
from replay_buffer import ReplayBuffer
from agent import Agent
from schedule_evaluator import evaluate_schedule
import time


USE_ACTION_MASKING = True
USE_RISK_PENALTY = True

# =====================================
# EXPERIMENT NAME
# =====================================

if USE_ACTION_MASKING and USE_RISK_PENALTY:

    EXPERIMENT_NAME = "Proposed"

elif USE_ACTION_MASKING and not USE_RISK_PENALTY:

    EXPERIMENT_NAME = "D3QN_Masking"

elif not USE_ACTION_MASKING and USE_RISK_PENALTY:

    EXPERIMENT_NAME = "D3QN_Risk"

else:

    EXPERIMENT_NAME = "D3QN_Plain"

# ---------------------------------------------------------------

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# =====================================================
# GREEDY FUNCTION
# =====================================================
def evaluate_greedy(
        agent,
        dax_path,
        use_risk_penalty,
        risk_weight,
        seed=None,
        vm_risks=None,
        risk_thresholds=None,
        use_security_mask=True,
        use_risk_mask=False,
        instance_path=None,
        budget_mode="medium"
):
    eval_env = WorkflowEnv(
        dax_path,
        use_risk_penalty=use_risk_penalty,
        risk_weight=risk_weight,
        seed=seed,
        vm_risks=vm_risks,
        risk_thresholds=risk_thresholds,
        use_security_mask=use_security_mask,
        use_risk_mask=use_risk_mask,
        instance_path=instance_path,
        budget_mode=budget_mode
    )

    state = eval_env.reset()

    done = False

    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    while not done:

        task_id = eval_env.get_current_task()

        task = eval_env.tasks[task_id]

        valid_actions = eval_env.get_valid_actions(
            task
        )

        action = agent.select_action(
            state,
            valid_actions
        )

        state, _, done = eval_env.step(
            action
        )

    agent.epsilon = old_epsilon
    return eval_env.makespan


def evaluate_policy(
        agent,
        dax_path,
        use_risk_penalty,
        risk_weight,
        seed,
        vm_risks=None,
        risk_thresholds=None,
        use_security_mask=True,
        use_risk_mask=False,
        instance_path=None,
        budget_mode="medium"
):
    eval_env = WorkflowEnv(
        dax_path,
        use_risk_penalty=use_risk_penalty,
        risk_weight=risk_weight,
        seed=seed,
        vm_risks=vm_risks,
        risk_thresholds=risk_thresholds,
        use_security_mask=use_security_mask,
        use_risk_mask=use_risk_mask,
        instance_path=instance_path,
        budget_mode=budget_mode
    )
    print("EVAL INSTANCE:", eval_env.instance_path)
    print("Evaluation risk weight =", eval_env.risk_weight)
    state = eval_env.reset()

    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    done = False

    runtime_start = time.perf_counter()

    steps = 0
    total_reward = 0

    while not done:

        steps += 1

        if steps > len(eval_env.tasks) * 10:
            print("Evaluation stopped: too many invalid/no-progress actions")
            break

        task_id = eval_env.get_current_task()

        if task_id is None:
            break

        task = eval_env.tasks[task_id]

        if agent.use_action_masking:

            valid_actions = eval_env.get_valid_actions(
                task
            )

        else:

            valid_actions = list(
                range(eval_env.action_size)
            )

        state, reward, done = eval_env.step(
            agent.select_action(
                state,
                valid_actions
            )
        )

        total_reward += reward

    runtime_ms = (time.perf_counter() - runtime_start) * 1000
    agent.epsilon = old_epsilon

    metrics = eval_env.evaluate_schedule()
    metrics["security_satisfaction"] = metrics["security_satisfaction_rate"]
    metrics["eval_seed"] = seed
    metrics["inference_runtime_ms"] = runtime_ms
    return metrics


# =====================================================
# TRAIN FUNCTION
# =====================================================

def train_model(
        dax_path="workflows/MONTAGE.dax",
        # dax_path="workflows/Epigenomics.dax",
        lr=0.001,
        gamma=0.99,



        batch_size=64,
        episodes=100,
        save_path="results/default",
        eval_seeds=(101, 202, 303, 404, 505),
        use_action_masking=None,
        use_risk_penalty=None,
        risk_weight=0.00,
        experiment_name=None,
        train_seed=42,
        vm_risks=None,
        risk_thresholds=None,
        use_security_mask=True,
        use_risk_mask=False,
        instance_path=None,
        evaluation_instance_paths=None,
        budget_mode="medium"
):
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(">>> ENTER train_model")
    dax_path = Path(dax_path)
    workflow_name = dax_path.stem  # MONTAGE_50
    # workflow_file = dax_path.name
    if use_action_masking is None:
        use_action_masking = USE_ACTION_MASKING

    if use_risk_penalty is None:
        use_risk_penalty = USE_RISK_PENALTY

    if experiment_name is None:

        if use_action_masking and use_risk_penalty:
            experiment_name = "Proposed"

        elif use_action_masking and not use_risk_penalty:
            experiment_name = "D3QN_Masking"

        elif not use_action_masking and use_risk_penalty:
            experiment_name = "D3QN_Risk"

        else:
            experiment_name = "D3QN_Plain"

    random.seed(train_seed)
    np.random.seed(train_seed)
    torch.manual_seed(train_seed)

    # ==========================================
    # CREATE SAVE DIRECTORY
    # ==========================================

    os.makedirs(
        save_path,
        exist_ok=True
    )

    # ==========================================
    # ENV
    # ==========================================

    env = WorkflowEnv(
        dax_path,
        use_risk_penalty=use_risk_penalty,
        risk_weight=risk_weight,
        seed=train_seed,
        vm_risks=vm_risks,
        risk_thresholds=risk_thresholds,
        use_security_mask=use_security_mask,
        use_risk_mask=use_risk_mask,
        instance_path=instance_path,
        budget_mode=budget_mode
    )
    print("TRAIN INSTANCE:", env.instance_path)
    state = env.reset()



    # ==========================================
    # INITIAL STATE TEST
    # ==========================================

    print("\nINITIAL STATE TEST")

    print("State:")
    print(state)

    print("State Length:", len(state))

    print("Min:", state.min())

    print("Max:", state.max())

    print("-" * 50)

    print("\n")
    print("=" * 60)
    print("EXPERIMENT:", experiment_name)
    print("MASKING:", use_action_masking)
    print("RISK PENALTY:", use_risk_penalty)
    print("RISK WEIGHT:", risk_weight)
    print("=" * 60)

    # ==========================================
    # DIMENSIONS
    # ==========================================

    state_dim = len(state)

    action_dim = env.action_size

    print("State Dim:", state_dim)

    print("Action Dim:", action_dim)

    print("-" * 50)

    # ==========================================
    # AGENT
    # ==========================================

    agent = Agent(

        state_dim,
        action_dim,

        lr=lr,
        gamma=gamma,
        batch_size=batch_size,

        use_action_masking = True  # resource feasibility is mandatory
    )

    # ==========================================
    # BUFFER
    # ==========================================

    buffer = ReplayBuffer()

    # ==========================================
    # BEST RESULT
    # ==========================================

    best_makespan = 1e9

    MODEL_PATH = os.path.join(
        save_path,
        f"{experiment_name}_{workflow_name}.pth"
    )

    # ==========================================
    # HISTORY
    # ==========================================

    reward_history = []
    security_history = []

    makespan_history = []

    cost_history = []

    risk_history = []

    violation_history = []

    invalid_history = []

    normalized_risk_history = []

    high_security_exposure_history = []

    budget_violation_history = []

    budget_excess_history =[]

    reward_makespan_history = []
    reward_cost_history = []
    reward_risk_history = []
    reward_budget_overflow_history = []

    epsilon_history = []

    # ==========================================
    # TRAIN LOOP
    # ==========================================

    for ep in range(1, episodes + 1):

        state = env.reset()

        done = False

        total_reward = 0

        step_count = 0

        vm_usage = []

        # ======================================
        # EPISODE LOOP
        # ======================================

        while not done:

            # ----------------------------------
            # ACTION
            # ----------------------------------

            task_id = env.get_current_task()

            task = env.tasks[task_id]

            if use_action_masking:

                valid_actions = env.get_valid_actions(
                    task
                )

            else:

                valid_actions = list(
                    range(env.action_size)
                )
            if ep == 235:
                print("TRAIN STATE")
                print(state[:20])

            action = agent.select_action(
                state,
                valid_actions
            )


            vm_usage.append(action)

            # ----------------------------------
            # ENV STEP
            # ----------------------------------

            next_state, reward, done = \
                env.step(action)

            if done:

                next_valid_actions = None

            elif use_action_masking:

                next_task_id = env.get_current_task()

                next_task = env.tasks[next_task_id]

                next_valid_actions = env.get_valid_actions(
                    next_task
                )

            else:

                next_valid_actions = list(
                    range(env.action_size)
                )

            # ----------------------------------
            # STATE CHECK
            # ----------------------------------

            if len(next_state) != state_dim:

                print("\nSTATE SIZE ERROR")

                print("Expected:", state_dim)

                print("Got:", len(next_state))

                break

            # ----------------------------------
            # SAVE TRANSITION
            # ----------------------------------

            buffer.push(

                state,
                action,
                reward,
                next_state,
                done,
                next_valid_actions
            )

            # ----------------------------------
            # TRAIN
            # ----------------------------------

            agent.train(buffer)

            # ----------------------------------
            # UPDATE
            # ----------------------------------

            state = next_state

            total_reward += reward

            step_count += 1

        # ======================================
        # TARGET UPDATE
        # ======================================
        # done in agent.py
        # if ep % 10 == 0:
        #
        #     agent.update_target()

        # ======================================
        # EPSILON DECAY
        # ======================================

        agent.decay_epsilon()

        # ======================================
        # METRICS
        # ======================================

        makespan = env.makespan

        env_metrics = env.evaluate_schedule()



        reward_history.append(
            total_reward
        )

        makespan_history.append(
            makespan
        )

        cost_history.append(
            env.total_cost
        )

        violation_history.append(
            env_metrics["violations"]
        )
        invalid_history.append(
            env.invalid_actions
        )

        normalized_risk_history.append(
            env.normalized_risk_exposure
        )

        high_security_exposure_history.append(
            env.high_security_exposure
        )

        budget_violation_history.append(
            env.budget_violation
        )
        budget_excess_history.append(
            env.budget_excess
        )

        reward_makespan_history.append(sum(env.reward_component_history["makespan"]))
        reward_cost_history.append(sum(env.reward_component_history["cost"]))
        reward_risk_history.append(sum(env.reward_component_history["risk"]))
        reward_budget_overflow_history.append(
            sum(env.reward_component_history["budget_overflow"])
        )

        epsilon_history.append(
            agent.epsilon
        )


        #____________________________________________________________________________________
        # --------------------------------------
        # SECURITY RATE
        # --------------------------------------

        security_rate = (

                                (
                                        len(env.tasks)
                                        - env_metrics["violations"]
                                )

                                / len(env.tasks)

                        ) * 100

        security_history.append(
            security_rate
        )

        # ________ files for different risk weight _______________
        # --------------------------------------
        # Sensitivity logging
        # --------------------------------------

        if "sensitivity_rows" not in locals():
            sensitivity_rows = []

        sensitivity_rows.append({
            "episode": ep,
            "risk_weight": risk_weight,
            "reward": total_reward,
            "makespan": makespan,
            "normalized_risk": env.normalized_risk_exposure,
            "security_satisfaction": security_rate,
            "reward_makespan": reward_makespan_history[-1],
            "reward_cost": reward_cost_history[-1],
            "reward_risk": reward_risk_history[-1],
            "reward_budget": reward_budget_overflow_history[-1],
        })

        # --------------------------------------
        # RISK
        # --------------------------------------

        risk_history.append(
            env.total_risk_exposure
        )

        # ======================================
        # TRAINING MONITOR ONLY
        # ======================================

        best_makespan = min(
            best_makespan,
            makespan
        )

        print(
            f"episode={ep}",
            f"train={makespan:.2f}",
            f"epsilon={agent.epsilon:.3f}"
        )
        # ======================================
        # STATS
        # ======================================

        avg_last_20 = np.mean(
            makespan_history[-20:]
        )

        vm_counter = Counter(vm_usage)

        if ep in [1, 50, 100, 200, 250, 300, 350, 400, 450, 500]:

            print("\nVM DETAILS")

            for vm_id, count in vm_counter.items():
                vm = env.vm_types[vm_id]

                print(
                    vm.name,
                    "| used:", count,
                    "| cost:", vm.cost,
                    "| risk:", vm.risk,
                    "| security:", vm.security_level
                )

        # ======================================
        # IMPORTANT EPISODES
        # ======================================

        if ep in [1, 50, 100, 200, 250, 300, 350, 400, 450, 500,1000,2000]:

            print("\n")
            print("=" * 60)

            print("EPISODE:", ep)

            print("-" * 60)

            print(
                "Reward:",
                round(total_reward, 2)
            )

            print(
                "Makespan:",
                round(makespan, 2)
            )

            print(
                "Best Training Makespan:",
                round(best_makespan, 2)
            )

            print(
                "Average Makespan (last20):",
                round(avg_last_20, 2)
            )

            print(
                "Total Cost:",
                round(env.total_cost, 2)
            )
            print(
                "Invalid Actions:",
                env.invalid_actions
            )

            print(
                "Final Security Violations:",
                env_metrics["violations"]
            )
            print(
                "Security Satisfaction:",
                round(security_rate, 2)
            )
            print(
                "Total Risk Exposure:",
                round(env.total_risk_exposure, 3)
            )

            print(
                "Epsilon:",
                round(agent.epsilon, 3)
            )
            print(
                "Total Risk Exposure:",
                round(env.total_risk_exposure, 3)
            )

            print(
                "Normalized Risk Exposure:",
                round(env.normalized_risk_exposure, 3)
            )

            print(
                "High Security Exposure:",
                round(env.high_security_exposure, 3)
            )

            print(
                "Budget Violation:",
                env.budget_violation
            )
            print(
                "Steps:",
                step_count
            )

            print(
                "VM Usage:",
                dict(vm_counter)
            )

            print("=" * 60)

    # ==========================================
    # TRAINING FINISHED
    # ==========================================

    print("\nTraining Finished")

    print(
        "Best Training Makespan:",
        round(best_makespan, 2)
    )

    checkpoint = {

        "episode": episodes,

        "training_best_makespan": best_makespan,

        "final_train_makespan": makespan_history[-1],

        "model_state_dict": agent.q_net.state_dict(),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "state_schema_version": env.STATE_SCHEMA_VERSION

    }

    torch.save(
        checkpoint,
        MODEL_PATH
    )

    print(
        "FINAL MODEL SAVED TO:",
        MODEL_PATH
    )

    # ==========================================
    # SAVE CSV
    # ==========================================

    results = pd.DataFrame({

        "episode": list(
            range(1, episodes + 1)
        ),

        "reward": reward_history,

        "makespan": makespan_history,

        "total_cost": cost_history,

        "security_satisfaction":
            security_history,

        "violations":
            violation_history,

        "invalid_actions":
            invalid_history,

        "total_risk":
            risk_history,

        "normalized_risk":
            normalized_risk_history,

        "high_security_exposure":
            high_security_exposure_history,

        "budget_violation":
            budget_violation_history,

        "budget_excess":
            budget_excess_history,

        "reward_makespan_component": reward_makespan_history,

        "reward_cost_component": reward_cost_history,

        "reward_risk_component": reward_risk_history,

        "reward_budget_overflow_component": reward_budget_overflow_history,

        "epsilon":
            epsilon_history



    })

    results.to_csv(

         os.path.join(
             save_path,
             f"{experiment_name}_{workflow_name}.csv"
         ),

        index=False
    )

    config = {
        "experiment_name": experiment_name,
        "dax_path": dax_path.name,
        "lr": lr,
        "gamma": gamma,
        "batch_size": batch_size,
        "episodes": episodes,
        "train_seed": train_seed,
        "eval_seeds": list(eval_seeds),
        "use_action_masking": use_action_masking,
        "use_security_mask": use_security_mask,
        "use_risk_mask": use_risk_mask,
        "use_risk_penalty": use_risk_penalty,
        "risk_weight": risk_weight,
        "risk_thresholds": env.risk_thresholds,
        "vm_risks_override": vm_risks,
        "risk_thresholds_override": risk_thresholds,
        "budget_limit": env.budget_limit,
        "budget_mode": budget_mode,
        "Cmin": env.Cmin,
        "Cmax": env.Cmax,
        "state_dim": state_dim,
        "state_schema_version": env.STATE_SCHEMA_VERSION,
        "vm_types": [
            {
                "vm_id": vm.vm_id,
                "name": vm.name,
                "vcpu": vm.vcpu,
                "ram": vm.ram,
                "cost": vm.cost,
                "risk": vm.risk,
                "security_level": vm.security_level
            }
            for vm in env.vm_types
        ],
        "tasks": [
            {
                "task_id": task.id,
                "runtime": task.runtime,
                "security": task.security,
                "cpu_req": task.cpu_req,
                "ram_req": task.ram_req
            }
            for task in env.tasks.values()
        ]
    }

    config_path = os.path.join(
        save_path,
        f"{experiment_name}_{workflow_name}_config.json"
    )

    with open(config_path, "w") as f:
        json.dump(
            config,
            f,
            indent=2
        )

    print(f"{experiment_name} saved")

    # ==========================================
    # FINAL EVALUATION
    # ==========================================

    print("\nStarting Final Evaluation...")

    checkpoint = torch.load(
        MODEL_PATH
    )

    if (checkpoint.get("state_dim") != state_dim
            or checkpoint.get("action_dim") != action_dim
            or checkpoint.get("state_schema_version") != env.STATE_SCHEMA_VERSION):
        raise ValueError("Checkpoint/environment state or action dimension mismatch")


    agent.q_net.load_state_dict(
        checkpoint["model_state_dict"]
    )

    agent.epsilon = 0.0

    evaluation_rows = []

    for seed in eval_seeds:
        evaluation_instance_path = (
            evaluation_instance_paths.get(seed, instance_path)
            if evaluation_instance_paths is not None
            else instance_path
        )
        metrics = evaluate_policy(
            agent,
            dax_path,
            use_risk_penalty,
            risk_weight,
            seed,
            vm_risks=vm_risks,
            risk_thresholds=risk_thresholds,
            use_security_mask=use_security_mask,
            use_risk_mask=use_risk_mask,
            instance_path=evaluation_instance_path,
            budget_mode=budget_mode
        )

        evaluation_rows.append(metrics)

        print("\nDEBUG METRICS")
        print(metrics)
        print("METRIC KEYS:", list(metrics.keys()))
        print("=" * 80)

        print(
            f"eval_seed={seed}",
            f"makespan={metrics['makespan']:.2f}",
            f"cost={metrics['total_cost']:.2f}",
            f"budget={metrics['budget_violation']}",
            f"violations={metrics['final_security_violations']}",
            f"risk={metrics['normalized_risk_exposure']:.3f}"
        )

    evaluation_results = pd.DataFrame(
        evaluation_rows
    )

    evaluation_path = (
        os.path.join(
            save_path,
            f"{experiment_name}_{workflow_name}_evaluation.csv"
        )
    )

    evaluation_results.to_csv(
        evaluation_path,
        index=False
    )

    print("\nFinal Evaluation Summary")
    print(
        evaluation_results[
            [
                "makespan",
        "total_cost",
        "security_satisfaction",
        "final_security_violations",
        "total_risk_exposure",
        "normalized_risk_exposure",
        "high_security_exposure",
        "normalized_high_security_exposure",
        "budget_violation",
        "invalid_actions"

            ]
        ].mean(numeric_only=True)


    )
    # =====================================================
    # FINAL METRICS (for thesis)
    # =====================================================

    avg = evaluation_results.mean(numeric_only=True)
    print("\n" + "=" * 42)
    print("FINAL AVERAGE RESULTS")
    print("=" * 42)

    print(
        f"Risk weight                 : "
        f"{risk_weight:.2f}"
    )

    print(
        f"Makespan                    : "
        f"{avg['makespan']:.2f}"
    )

    print(
        f"Total Cost                 : "
        f"{avg['total_cost']:.2f}"
    )

    print(
        f"Security Satisfaction (%)  : "
        f"{avg['security_satisfaction']:.2f}"
    )

    print(
        f"Security Violations        : "
        f"{avg['final_security_violations']:.2f}"
    )

    print(
        f"Total Risk Exposure        : "
        f"{avg['total_risk_exposure']:.4f}"
    )

    print(
        f"Normalized Risk Exposure   : "
        f"{avg['normalized_risk_exposure']:.4f}"
    )

    print(
        f"High Security Exposure     : "
        f"{avg['high_security_exposure']:.4f}"
    )

    print(
        f"Normalized High-Sec Risk   : "
        f"{avg['normalized_high_security_exposure']:.4f}"
    )

    print(
        f"Budget Violation           : "
        f"{avg['budget_violation']:.2f}"
    )

    print(
        f"Invalid Actions            : "
        f"{avg['invalid_actions']:.2f}"
    )

    print("=" * 42)
    del agent
    del buffer
    torch.cuda.empty_cache()

    #___________________differetnt risk weight___________
    pd.DataFrame(sensitivity_rows).to_csv(
        os.path.join(
            save_path,
            f"{experiment_name}_{workflow_name}_risk_weight_{risk_weight:.2f}_training.csv"
        ),
        index=False
    )

    # ==========================================
    # RETURN RESULTS
    # ==========================================

    return evaluation_results



# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    print(">>> MAIN STARTED <<<")
    # train_model(risk_weight=1)
    # for rw in [0.0, 0.3, 0.6, 1.0]:
    #     print("\n")
    #     print("=" * 70)
    #     print(f"\n######## risk_weight = {rw} ########")
    #     print(f"Running Experiment (risk_weight={rw})")
    #     print("=" * 70)

    # rw =  0.8 # مقدار موردنظر

    train_model(
            risk_weight=0.00,
            episodes=500,
            save_path=f"results/test/risk/Montage_0.00"
        )
    # for rw in [0.00, 0.03, 0.05, 0.07, 0.10]:
    #     train_model(
    #         risk_weight=rw,
    #         episodes=500,
    #         save_path=f"results/test/risk_weight_{rw:.2f}"
    #     )

    # print(f"==== END {rw} ====")


