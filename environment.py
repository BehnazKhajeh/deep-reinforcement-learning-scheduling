from __future__ import annotations

import copy
import json
from pathlib import Path
import numpy as np

from env_dag_parser import parse_dax
from ranku import compute_ranku
from vm_config import VM_TYPES


class WorkflowEnv:
    BILLING_UNIT = 10.0
    STATE_SCHEMA_VERSION = 3

    def __init__(self, dax_path, *, instance_path=None, seed=0, risk_weight=0.0,
                 use_security_mask=True, use_risk_mask=False, use_risk_penalty=False,
                 budget_mode="medium", vm_risks=None, risk_thresholds=None):
        self.dax_path = Path(dax_path).resolve()
        self.instance_path = Path(instance_path).resolve() if instance_path else self.dax_path.with_suffix(".task_instance.json")
        self.seed = seed
        self.risk_weight = risk_weight
        self.use_security_mask = use_security_mask
        self.use_risk_mask = use_risk_mask
        self.use_risk_penalty = use_risk_penalty
        self.budget_mode = budget_mode
        self.risk_thresholds = risk_thresholds or {1: 1.0, 2: 0.7, 3: 0.3}
        self.G, self.tasks = parse_dax(self.dax_path, self.instance_path, seed)
        self.ranku = compute_ranku(self.G, self.tasks)
        self.vm_types = copy.deepcopy(VM_TYPES)
        if vm_risks:
            for vm in self.vm_types:
                if vm.vm_id in vm_risks:
                    vm.risk = float(vm_risks[vm.vm_id])
        self.num_vms = len(self.vm_types)
        self.action_size = self.num_vms
        self.Cmin, self.Cmedium, self.Cmax = self._compute_budget_limits()
        self.budget_limit = {"low": self.Cmin, "medium": self.Cmedium, "high": self.Cmax}[budget_mode]
        self.state_size = self.num_vms * 8 + 7 + 4
        self.reset()

    def _compute_budget_limits(self):
        low_duration = {vm.vm_id: 0.0 for vm in self.vm_types}
        fast_duration = {vm.vm_id: 0.0 for vm in self.vm_types}
        for task in self.tasks.values():
            allowed = [vm for vm in self.vm_types if vm.total_cpu >= task.cpu_req and vm.total_ram >= task.ram_req and vm.security_level >= task.security]
            if not allowed:
                raise ValueError(f"No security/resource-feasible VM for {task.id}")
            cheap = min(allowed, key=lambda vm: (vm.cost, vm.vm_id))
            speedy = min(allowed, key=lambda vm: (vm.runtime(task), vm.vm_id))
            low_duration[cheap.vm_id] += cheap.runtime(task)
            fast_duration[speedy.vm_id] += speedy.runtime(task)
        low = sum(vm.billed_cost_for_duration(low_duration[vm.vm_id], self.BILLING_UNIT) for vm in self.vm_types)
        fast = sum(vm.billed_cost_for_duration(fast_duration[vm.vm_id], self.BILLING_UNIT) for vm in self.vm_types)
        cmin = max(low, 1e-6)
        cmax = max(cmin, fast)
        return cmin, (cmin + cmax) / 2.0, cmax

    def reset(self):
        for vm in self.vm_types: vm.reset_schedule()
        self.task_start, self.task_finish, self.task_vm = {}, {}, {}
        self.done_tasks = set()
        self.ready_tasks = self.get_ready_tasks()
        self.makespan = self.total_cost = 0.0
        self.steps = 0
        self.invalid_actions = 0
        self.security_level_violations = self.risk_threshold_violations = 0
        self.violating_tasks = set()
        self.final_security_violations = 0
        self.total_risk_exposure = self.sensitivity_weighted_risk_exposure = 0.0
        self.high_security_task_exposure = 0.0
        self.high_security_exposure = 0.0  # compatibility alias
        self.normalized_risk_exposure = 0.0
        self.security_satisfaction = 100.0
        self.budget_excess = 0.0
        self.budget_violation = 0
        self.reward_component_history = {
            "makespan": [], "cost": [], "risk": [], "budget_overflow": []
        }
        return self.get_state()

    def get_ready_tasks(self):
        return sorted(tid for tid, task in self.tasks.items() if tid not in self.done_tasks and all(p in self.done_tasks for p in task.parents))

    def get_current_task(self):
        return self.ready_tasks[0] if self.ready_tasks else None

    def _resource_valid(self, task, vm):
        return task.cpu_req <= vm.total_cpu and task.ram_req <= vm.total_ram

    def get_valid_actions(self, task):
        valid = []
        for vm in self.vm_types:
            if not self._resource_valid(task, vm):
                continue  # resource mask is mandatory for every variant
            if self.use_security_mask and vm.security_level < task.security:
                continue
            if self.use_risk_mask and vm.risk > self.risk_thresholds[task.security]:
                continue
            valid.append(vm.vm_id)
        return valid

    def get_state(self):
        tid = self.get_current_task()
        if tid is None: return np.zeros(self.state_size, dtype=np.float32)
        task = self.tasks[tid]
        max_rt = max(t.runtime for t in self.tasks.values())
        horizon = max_rt * len(self.tasks) + 1e-9
        max_cpu, max_ram = max(v.total_cpu for v in self.vm_types), max(v.total_ram for v in self.vm_types)
        max_cost = max(v.cost for v in self.vm_types)
        state = []
        for vm in self.vm_types:
            parent_ready = max(
                (self.task_finish[p] for p in task.parents),
                default=0.0
            )

            runtime = vm.runtime(task)

            candidate_start = vm.find_earliest_slot(
                task,
                parent_ready,
                runtime
            )

            candidate_finish = candidate_start + runtime

            cpu, ram = vm._usage_at(candidate_start)
            resource_status = min(
                (vm.total_cpu - cpu) / vm.total_cpu,
                (vm.total_ram - ram) / vm.total_ram,
            )

            state += [
                candidate_start / horizon,
                candidate_finish / horizon,
                resource_status,
                vm.security_level / 3.,
                vm.risk,
                vm.cost / max_cost,
                vm.total_cpu / max_cpu,
                vm.total_ram / max_ram,
            ]
        state += [task.runtime/max_rt, task.cpu_req/max_cpu, task.ram_req/max_ram, task.security/3.,
                  self.ranku[tid]/max(self.ranku.values()), len(task.parents)/max(1,len(self.tasks)), len(task.children)/max(1,len(self.tasks))]
        state += [len(self.done_tasks)/len(self.tasks), self.makespan/horizon,
                  self.total_cost/max(self.budget_limit,1e-9), max(0.,self.budget_limit-self.total_cost)/max(self.budget_limit,1e-9)]
        assert len(state) == self.state_size
        return np.asarray(state, dtype=np.float32)

    def _final_violation(self, task, vm):
      ...
        return security, risk

    def step(self, vm_id):
        self.steps += 1
        tid = self.get_current_task()
        if tid is None: return self.get_state(), 0.0, True
        task = self.tasks[tid]
        if vm_id not in self.get_valid_actions(task):
            self.invalid_actions += 1
            return self.get_state(), -5.0, False
        vm = self.vm_types[vm_id]
        parent_ready = max((self.task_finish[p] for p in task.parents), default=0.0)
        runtime = vm.runtime(task)
        start = vm.find_earliest_slot(task, parent_ready, runtime)
        finish, old_makespan, old_cost = start + runtime, self.makespan, self.total_cost
        vm.allocate(task, start, finish)
        self.task_start[tid], self.task_finish[tid], self.task_vm[tid] = start, finish, vm_id
        self.done_tasks.add(tid); self.ready_tasks = self.get_ready_tasks()
        self.makespan = max(self.task_finish.values())
        self.total_cost = self.compute_total_cost()
        self.budget_violation = int(self.total_cost > self.budget_limit)
        sec_bad, risk_bad = self._final_violation(task, vm)
        self.security_level_violations += int(sec_bad); self.risk_threshold_violations += int(risk_bad)
        # if ... 
          #makespan_penalty 
          #cost_penalty 
          #risk_penalty 
        overflow = max(0., self.total_cost-self.budget_limit) / max(self.budget_limit,1e-9)
        risk_component = self.risk_weight * risk_penalty
        budget_overflow_component = 5. * overflow
        self.reward_component_history["makespan"].append(makespan_penalty)
        self.reward_component_history["cost"].append(cost_penalty)
        self.reward_component_history["risk"].append(risk_component)
        self.reward_component_history["budget_overflow"].append(budget_overflow_component)
           # reward 
        return self.get_state(), reward, len(self.done_tasks) == len(self.tasks)

    def compute_total_cost(self):
        return sum(vm.billed_cost(self.BILLING_UNIT) for vm in self.vm_types)

    def evaluate_schedule(self):
        n = max(1, len(self.task_vm)); total_security = self.final_security_violations
        return {"completed": len(self.done_tasks)==len(self.tasks), "scheduled_tasks": len(self.task_vm), "total_tasks": len(self.tasks), "steps": self.steps,
                "makespan": self.makespan, "total_cost": self.total_cost,
                "budget_violation": int(self.total_cost > self.budget_limit), "invalid_actions": self.invalid_actions,
                "security_level_violations": self.security_level_violations, "risk_threshold_violations": self.risk_threshold_violations,
                "total_security_violations": total_security, "final_security_violations": total_security,
                "security_satisfaction_rate": 100*(1-total_security/n), "total_risk_exposure": self.total_risk_exposure,
                "mean_risk_exposure": self.total_risk_exposure/n, "sensitivity_weighted_risk_exposure": self.sensitivity_weighted_risk_exposure,
                "high_security_task_exposure": self.high_security_task_exposure,
                "normalized_risk_exposure": self.sensitivity_weighted_risk_exposure/max(1,3*n), "budget_limit": self.budget_limit,
                "Cmin": self.Cmin, "Cmax": self.Cmax,
                # Backward-compatible column aliases used by the training CSV.
                "violations": total_security, "security_satisfaction": 100*(1-total_security/n),
                "total_risk": self.total_risk_exposure, "normalized_risk": self.sensitivity_weighted_risk_exposure/max(1,3*n),
                "high_security_exposure": self.high_security_task_exposure,
                "normalized_high_security_exposure": self.high_security_task_exposure/max(1, sum(t.security == 3 for t in self.tasks.values()))}

    def save_instance_metadata(self, path):
        Path(path).write_text(json.dumps({"state_dim": self.state_size, "budget": {"Cmin":self.Cmin,"Cmedium":self.Cmedium,"Cmax":self.Cmax}, "tasks": {k:{"security":t.security,"cpu":t.cpu_req,"ram":t.ram_req} for k,t in self.tasks.items()}}, indent=2))
