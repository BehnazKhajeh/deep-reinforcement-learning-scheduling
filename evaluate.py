def evaluate(agent, env):
    print("Evaluation started")

    state = env.reset()

    print("Reset done")

    done = False



    counter = 0
    previous_done_tasks = -1

    while not done:

        counter += 1

        if counter % 20 == 0:
            print(
                "step=", counter,
                "current_task=", env.get_current_task(),
                "done_tasks=", len(env.done_tasks)
            )

        if counter > 1000:
            print("STUCK")
            print("current_task =", env.get_current_task())
            print("done_tasks =", len(env.done_tasks))

            break

        task_id = env.get_current_task()

        task = env.tasks[task_id]

        valid_actions = env.get_valid_actions(task)

        action = agent.select_greedy_action(
            state,
            valid_actions
        )

        if counter % 10 == 0:

            print(
                f"\nSTEP {counter}"
            )

            print(
                f"Current Task: {task_id}"
            )

            print(
                f"Done Tasks: {len(env.done_tasks)} / {len(env.tasks)}"
            )

            print(
                f"Valid Actions: {valid_actions}"
            )

            print(
                f"Selected Action: {action}"
            )

        if counter > 500:

            print("\n===================")
            print("EVALUATION STUCK")
            print("===================")

            print(
                f"Current Task: {task_id}"
            )

            print(
                f"Done Tasks: {len(env.done_tasks)} / {len(env.tasks)}"
            )

            print(
                f"Valid Actions: {valid_actions}"
            )

            print(
                f"Selected Action: {action}"
            )

            break

        old_done_tasks = len(env.done_tasks)

        state, _, done = env.step(action)

        new_done_tasks = len(env.done_tasks)

        if new_done_tasks == old_done_tasks:

            print(
                f"WARNING: No progress at step {counter}"
            )

            print(
                f"Task {task_id}"
            )

            print(
                f"Action {action}"
            )

    return {

        "makespan":
            env.makespan,

        "cost":
            env.total_cost,

        "budget_violation":
            env.budget_violation,

        "security_violations":
            env.final_security_violations,

        "invalid_actions":
            env.invalid_actions,

        "security_satisfaction":
            env.security_satisfaction,

        "total_risk":
            env.total_risk_exposure,

        "normalized_risk":
            env.normalized_risk_exposure,

        "high_security_exposure":
            env.high_security_exposure
    }
