import argparse
import os
from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np
import random
import torch
from gymnasium import spaces
from sumo_rl import SumoEnvironment
from sumo_rl.environment.observations import ObservationFunction

from Sumo_RL import Agent as BaseAgent
from Sumo_RL import load_hyperparameters
from Sumo_RL import PHASE_MAP, SumoDQN, device, get_current_phase_from_obs
from experience_replay_sumo import ReplayMemory


SCRIPT_DIR = Path(__file__).resolve().parent
PHASE_WAIT_CAP = 120.0

PER_LANE_PHASE_MAP = {
    0: "Phase 1: North-South Through",
    1: "Phase 2: North-South Protected Left",
    2: "Phase 3: East-West Through",
    3: "Phase 4: East-West Protected Left",
}


def _lane_phase_index(lane_id: str) -> int:
    if lane_id.startswith(("n_t", "s_t")) and lane_id.endswith("_0"):
        return 0  # North-South through
    if lane_id.startswith(("n_t", "s_t")) and lane_id.endswith("_1"):
        return 1  # North-South left
    if lane_id.startswith(("e_t", "w_t")) and lane_id.endswith("_0"):
        return 2  # East-West through
    if lane_id.startswith(("e_t", "w_t")) and lane_id.endswith("_1"):
        return 3  # East-West left
    return 0


def better_reward(ts):
    current_queue = float(ts.get_total_queued())

    total_wait = 0.0
    for veh_id in ts.sumo.vehicle.getIDList():
        total_wait += ts.sumo.vehicle.getWaitingTime(veh_id)

    current_total_vehicles = ts.sumo.simulation.getMinExpectedNumber()

    if not hasattr(ts, "_prev_total_vehicles"):
        ts._prev_total_vehicles = current_total_vehicles

    cleared = max(0, ts._prev_total_vehicles - current_total_vehicles)
    ts._prev_total_vehicles = current_total_vehicles

    phase_demand = [0.0, 0.0, 0.0, 0.0]
    phase_wait = [0.0, 0.0, 0.0, 0.0]
    for lane_id, queue_value in zip(ts.lanes, ts.get_lanes_queue()):
        phase_idx = _lane_phase_index(lane_id)
        phase_demand[phase_idx] += float(queue_value)

        lane_wait_time = 0.0
        for veh_id in ts.sumo.lane.getLastStepVehicleIDs(lane_id):
            lane_wait_time += ts.sumo.vehicle.getWaitingTime(veh_id)
        phase_wait[phase_idx] += min(1.0, lane_wait_time / PHASE_WAIT_CAP)

    phase_demand = [min(1.0, value) for value in phase_demand]
    phase_wait = [min(1.0, value) for value in phase_wait]

    current_phase = int(ts.green_phase)
    served_pressure = phase_demand[current_phase] + phase_wait[current_phase]
    best_pressure = max(
        demand + wait for demand, wait in zip(phase_demand, phase_wait)
    )
    unserved_pressure_penalty = max(0.0, best_pressure - served_pressure)

    reward = (
        -0.001 * total_wait
        -0.1 * current_queue
        + 0.5 * cleared
        -0.05 * current_total_vehicles
        -1.0 * unserved_pressure_penalty
    )

    nominal_seconds = getattr(ts.env, "_nominal_num_seconds", None)
    if (
        nominal_seconds is not None
        and getattr(ts.env, "sim_step", 0) >= nominal_seconds
        and current_total_vehicles > 0
    ):
        reward -= 1.0

    return float(reward)


class PerLanePhaseObservationFunction(ObservationFunction):
    """
    Observation layout (21 dims total):
    - 4 current-phase one-hot
    - 1 min_green flag
    - 4 normalized phase-demand totals
    - 4 normalized phase-wait totals
    - 8 per-lane normalized queues

    This keeps the queue slice at [13:21], so the inherited training code
    can still compute queue summaries without modification.
    """

    def _lane_phase_index(self, lane_id: str) -> int:
        return _lane_phase_index(lane_id)

    def _lane_wait_normalized(self, lane_id: str) -> float:
        wait_time = 0.0
        for veh_id in self.ts.sumo.lane.getLastStepVehicleIDs(lane_id):
            wait_time += self.ts.sumo.vehicle.getWaitingTime(veh_id)
        return min(1.0, wait_time / PHASE_WAIT_CAP)

    def __call__(self) -> np.ndarray:
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]

        lane_queue = self.ts.get_lanes_queue()
        phase_demand = [0.0, 0.0, 0.0, 0.0]
        phase_wait = [0.0, 0.0, 0.0, 0.0]

        for lane_id, queue_value in zip(self.ts.lanes, lane_queue):
            phase_idx = self._lane_phase_index(lane_id)
            phase_demand[phase_idx] += float(queue_value)
            phase_wait[phase_idx] += self._lane_wait_normalized(lane_id)

        phase_demand = [min(1.0, value) for value in phase_demand]
        phase_wait = [min(1.0, value) for value in phase_wait]

        observation = np.array(
            phase_id + min_green + phase_demand + phase_wait + lane_queue,
            dtype=np.float32,
        )
        return observation

    def observation_space(self) -> spaces.Box:
        return spaces.Box(
            low=np.zeros(21, dtype=np.float32),
            high=np.ones(21, dtype=np.float32),
            dtype=np.float32,
        )


class Agent(BaseAgent):
    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)
        self._set_run_artifact_paths(
            os.path.join(self.runs_dir, f"sumo1_per_lane_{self.timestamp}.pt")
        )
        self.no_progress_patience_seconds = float("inf")
        self.debug_interval_seconds = 8.0

    def _get_step_total_co2_emission(self, env):
        return sum(
            env.sumo.vehicle.getCO2Emission(veh_id)
            for veh_id in env.sumo.vehicle.getIDList()
        ) * float(self.delta_time)

    def _baseline_has_co2_metrics(self, baseline):
        return (
            isinstance(baseline, dict)
            and "total_co2_emission" in baseline
            and "avg_co2_per_vehicle" in baseline
        )

    def _format_named_values(self, names, values, formatter):
        return ", ".join(
            f"{name}={formatter(value)}" for name, value in zip(names, values)
        )

    def _get_directional_queue_raw(self, env):
        traffic_signal = env.traffic_signals[env.ts_ids[0]]
        ns_queue_raw = 0.0
        we_queue_raw = 0.0
        for lane_id in traffic_signal.lanes:
            lane_queue = float(env.sumo.lane.getLastStepHaltingNumber(lane_id))
            if lane_id.startswith(("n_", "s_")):
                ns_queue_raw += lane_queue
            elif lane_id.startswith(("e_", "w_")):
                we_queue_raw += lane_queue
        return ns_queue_raw, we_queue_raw

    def _write_directional_queue_outputs(self, csv_path, png_path, queue_rows, title):
        with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["step_index", "sim_step", "ns_queue_raw", "we_queue_raw"])
            writer.writerows(queue_rows)

        ns_values = [row[2] for row in queue_rows]
        we_values = [row[3] for row in queue_rows]

        plt.figure(figsize=(9, 5))
        bins = range(0, int(max(ns_values + we_values + [1])) + 2)
        plt.hist(ns_values, bins=bins, density=True, alpha=0.35, color="red", label=f"NS, mean = {np.mean(ns_values):.2f}")
        plt.hist(we_values, bins=bins, density=True, alpha=0.35, color="royalblue", label=f"WE, mean = {np.mean(we_values):.2f}")
        plt.xlabel("Queue length bins")
        plt.ylabel("Normalized Frequency")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(png_path, dpi=150)
        plt.close()

    def _log_overtime_debug_snapshot(self, env, observation, prefix):
        traffic_signal = env.traffic_signals[env.ts_ids[0]]
        lane_names = list(traffic_signal.lanes)
        lane_vehicle_counts = [
            env.sumo.lane.getLastStepVehicleNumber(lane_name) for lane_name in lane_names
        ]
        lane_wait_times = [
            sum(env.sumo.vehicle.getWaitingTime(veh_id) for veh_id in env.sumo.lane.getLastStepVehicleIDs(lane_name))
            for lane_name in lane_names
        ]

        phase_idx = get_current_phase_from_obs(observation)
        phase_one_hot = observation[:4]
        min_green = observation[4]
        phase_demand = observation[5:9]
        phase_wait = observation[9:13]
        lane_queue = observation[13:21]

        self.log(
            f"{prefix} overtime debug | Sim step: {float(env.sim_step):.1f} | "
            f"Phase: {phase_idx} -> {PER_LANE_PHASE_MAP.get(phase_idx, 'Unknown Phase')} | "
            f"Min green: {min_green:.0f}"
        )
        self.log(f"{prefix} phase one-hot      : {np.array2string(phase_one_hot, precision=3)}")
        self.log(f"{prefix} phase demand       : {np.array2string(phase_demand, precision=3)}")
        self.log(f"{prefix} phase wait         : {np.array2string(phase_wait, precision=3)}")
        self.log(
            f"{prefix} lane queue norm     : "
            f"{self._format_named_values(lane_names, lane_queue, lambda value: f'{float(value):.3f}')}"
        )
        self.log(
            f"{prefix} lane vehicles raw   : "
            f"{self._format_named_values(lane_names, lane_vehicle_counts, lambda value: str(int(value)))}"
        )
        self.log(
            f"{prefix} lane waiting raw    : "
            f"{self._format_named_values(lane_names, lane_wait_times, lambda value: f'{float(value):.1f}s')}"
        )

    def create_env(self, route_file, use_gui=False):
        env = SumoEnvironment(
            net_file=self.net_file,
            route_file=route_file,
            use_gui=use_gui,
            num_seconds=self.num_seconds + self.clearance_extra_seconds,
            delta_time=self.delta_time,
            yellow_time=self.yellow_time,
            min_green=self.min_green,
            single_agent=True,
            reward_fn=better_reward,
            observation_class=PerLanePhaseObservationFunction,
            sumo_warnings=self.sumo_warnings,
            additional_sumo_cmd=self.additional_sumo_cmd if use_gui else "",
        )
        env._nominal_num_seconds = self.num_seconds
        return env

    def run_single_episode(self, env, controller="rl"):
        observation, info = env.reset()
        done = False
        truncated = False
        total_reward = 0.0
        empty_counter = 0
        overtime_logged = False
        next_debug_step = float(self.num_seconds)

        total_queue_normalized_sum = 0.0
        total_queue_raw_sum = 0.0
        total_waiting_time = 0.0
        total_co2_emission = 0.0
        vehicles_cleared = 0
        step_counter = 0
        prev_total_vehicles = None

        fixed_state = {"last_phase": None}

        while not done:
            state = torch.tensor(observation, dtype=torch.float32, device=device)

            if controller == "rl":
                action = self.select_action(
                    state_tensor=state,
                    observation=observation,
                    env=env,
                    epsilon=0.0,
                    is_training=False
                )
            elif controller == "fixed":
                action = self.select_fixed_action(observation, fixed_state)
            else:
                raise ValueError("controller must be 'rl' or 'fixed'")

            next_observation, reward, done, truncated, info = env.step(action)

            current_queue_normalized = float(sum(next_observation[13:21]))
            total_queue_normalized_sum += current_queue_normalized

            traffic_signal = env.traffic_signals[env.ts_ids[0]]
            current_queue_raw = float(traffic_signal.get_total_queued())
            total_queue_raw_sum += current_queue_raw

            waiting_time = 0.0
            for veh_id in env.sumo.vehicle.getIDList():
                waiting_time += env.sumo.vehicle.getWaitingTime(veh_id)
            total_waiting_time += waiting_time
            total_co2_emission += self._get_step_total_co2_emission(env)

            current_total_vehicles = env.sumo.simulation.getMinExpectedNumber()
            if prev_total_vehicles is not None:
                vehicles_cleared += max(0, prev_total_vehicles - current_total_vehicles)
            prev_total_vehicles = current_total_vehicles

            if env.sim_step >= self.num_seconds and current_total_vehicles > 0 and not overtime_logged:
                overtime_logged = True
                self.log(
                    f"Episode exceeded nominal duration {self.num_seconds}s and is continuing "
                    f"until clearance or hard cap. Vehicles left: {current_total_vehicles}"
                )

            if env.sim_step >= self.num_seconds and current_total_vehicles > 0 and float(env.sim_step) >= next_debug_step:
                self._log_overtime_debug_snapshot(env, next_observation, "Validation")
                next_debug_step += self.debug_interval_seconds

            step_counter += 1

            if self._traffic_is_finished(env):
                empty_counter += 1
            else:
                empty_counter = 0

            if truncated and current_total_vehicles > 0:
                truncated = False

            terminal_event = done or truncated or empty_counter >= self.early_stop_patience
            if terminal_event:
                reward = self._apply_terminal_remaining_penalty(reward, current_total_vehicles)

            total_reward += reward

            if empty_counter >= self.early_stop_patience:
                break

            observation = next_observation

        avg_queue = total_queue_normalized_sum / max(1, step_counter)
        avg_wait_per_vehicle = total_waiting_time / max(1, vehicles_cleared)
        avg_co2_per_vehicle = total_co2_emission / max(1, vehicles_cleared)
        remaining_vehicles = self._get_remaining_vehicles(env)

        return {
            "total_reward": total_reward,
            "avg_queue": avg_queue,
            "total_waiting_time": total_waiting_time,
            "total_co2_emission": total_co2_emission,
            "vehicles_cleared": vehicles_cleared,
            "avg_wait_per_vehicle": avg_wait_per_vehicle,
            "avg_co2_per_vehicle": avg_co2_per_vehicle,
            "simulation_steps": step_counter,
            "remaining_vehicles": remaining_vehicles,
        }

    def train(self, resume_from=None, unseen_only=False, target_episodes=None):
        fixed_baseline = None
        resume_checkpoint = None
        resume_log_file = None

        if resume_from is not None:
            resume_model_file = self._resolve_resume_model_file(resume_from)
            resume_checkpoint = self._load_checkpoint_data(resume_model_file)
            resume_log_file = str(Path(resume_model_file).with_suffix(".log"))
            self.log(f"Resuming training from checkpoint: {resume_model_file}")
            self.log("Resume mode is writing to a new run. Source checkpoint files will remain unchanged.")

        self.log("Starting training...")
        self.log(f"Best checkpoint path : {self.best_model_file}")
        self.log(f"Final checkpoint path: {self.final_model_file}")
        self.log(f"Run log path         : {self.log_file}")
        self.log(f"Run plot path        : {self.graph_file}")

        all_routes = sorted(self._get_test_routes() + [])
        all_routes = sorted(
            [
                os.path.join(self.route_dir, file_name)
                for file_name in os.listdir(self.route_dir)
                if file_name.endswith(".rou.xml")
            ]
        )
        rng = random.Random(self.hp["random_seed"])
        rng.shuffle(all_routes)

        n_routes = len(all_routes)
        train_end = int(0.70 * n_routes)
        val_end = int(0.85 * n_routes)

        train_routes = all_routes[:train_end]
        val_routes = all_routes[train_end:val_end]
        test_routes = all_routes[val_end:]

        if len(val_routes) == 0:
            raise ValueError("Validation split is empty. Add more routes or change the split.")

        self.val_routes = val_routes[:self.validation_routes_count]
        self.test_routes = test_routes

        if resume_from is not None and unseen_only:
            source_log_file = resume_log_file if resume_log_file is not None else self.log_file
            seen_route_names = self._extract_seen_training_routes_from_log(source_log_file)
            train_routes = [
                route_file
                for route_file in train_routes
                if os.path.basename(route_file) not in seen_route_names
            ]
            self.log(f"Seen training routes from log: {len(seen_route_names)}")
            self.log(f"Remaining unseen training routes: {len(train_routes)}")
            if len(train_routes) == 0:
                self.log("No unseen training routes remain for this run.")
                return

        self.log(f"Train routes: {len(train_routes)}")
        self.log(f"Validation routes: {len(self.val_routes)}")
        self.log(f"Test routes: {len(test_routes)}")

        temp_route = train_routes[0]
        temp_env = self.create_env(temp_route, use_gui=False)
        initial_observation, _ = temp_env.reset()

        state_dim = len(initial_observation)
        action_dim = temp_env.action_space.n
        temp_env.close()

        self.log(f"State dimension: {state_dim}")
        self.log(f"Action dimension: {action_dim}")

        self.policy_dqn = SumoDQN(state_dim, action_dim, self.fc1_nodes).to(device)
        self.target_dqn = SumoDQN(state_dim, action_dim, self.fc1_nodes).to(device)
        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())

        self.optimizer = torch.optim.Adam(self.policy_dqn.parameters(), lr=self.learning_rate_a)
        self.memory = ReplayMemory(self.replay_memory_size)

        if resume_checkpoint is None:
            epsilon = self.epsilon_init
            step_count = 0
            rewards_per_episode = []
            epsilon_history = []
            validation_wait_history = []
            best_validation_wait = float("inf")
        else:
            policy_state_dict = resume_checkpoint.get("policy_state_dict")
            if policy_state_dict is not None:
                self.policy_dqn.load_state_dict(policy_state_dict)

            target_state_dict = resume_checkpoint.get("target_state_dict")
            if target_state_dict is not None:
                self.target_dqn.load_state_dict(target_state_dict)
            else:
                self.target_dqn.load_state_dict(self.policy_dqn.state_dict())

            optimizer_state_dict = resume_checkpoint.get("optimizer_state_dict")
            if optimizer_state_dict is not None:
                self.optimizer.load_state_dict(optimizer_state_dict)

            memory_items = resume_checkpoint.get("memory")
            if memory_items:
                self.memory.memory.extend(memory_items)

            epsilon = resume_checkpoint.get("epsilon", self.epsilon_init)
            step_count = resume_checkpoint.get("step_count", 0)
            rewards_per_episode = list(resume_checkpoint.get("rewards_per_episode", []))
            epsilon_history = list(resume_checkpoint.get("epsilon_history", []))
            validation_wait_history = list(resume_checkpoint.get("validation_wait_history", []))
            best_validation_wait = resume_checkpoint.get("best_validation_wait", float("inf"))
            episode = resume_checkpoint.get("episode", 0)
            fixed_baseline = resume_checkpoint.get("fixed_baseline")

            self.log(f"Recovered progress up to episode {episode} with epsilon {epsilon:.4f}.")
            if resume_checkpoint.get("optimizer_state_dict") is None:
                self.log(
                    "Legacy checkpoint detected: model weights were restored, "
                    "but optimizer and replay memory were not available."
                )

        if resume_checkpoint is None:
            episode = 0

        if fixed_baseline is None or not self._baseline_has_co2_metrics(fixed_baseline):
            if fixed_baseline is not None:
                self.log(
                    "Stored fixed validation baseline is missing CO2 metrics. "
                    "Recomputing the fixed baseline for this resumed run."
                )
            fixed_baseline = self.evaluate_controller_on_routes(self.val_routes, controller="fixed")

        self.log("========== FIXED CONTROLLER VALIDATION BASELINE ==========")
        self.log(f"Validation Avg Queue        : {fixed_baseline['avg_queue']:.4f}")
        self.log(f"Validation Total Waiting   : {fixed_baseline['total_waiting_time']:.4f}")
        self.log(f"Validation Total CO2       : {fixed_baseline['total_co2_emission']:.4f} mg")
        self.log(f"Validation Vehicles Cleared: {fixed_baseline['vehicles_cleared']:.4f}")
        self.log(f"Validation Avg Wait/Vehicle: {fixed_baseline['avg_wait_per_vehicle']:.4f}")
        self.log(f"Validation Avg CO2/Vehicle : {fixed_baseline['avg_co2_per_vehicle']:.4f} mg")
        self.log("=========================================================")

        target_episode_limit = target_episodes if target_episodes is not None else self.num_episodes
        if unseen_only:
            unseen_target_limit = episode + len(train_routes)
            target_episode_limit = min(target_episode_limit, unseen_target_limit)
            self.log(f"Resume unseen-only target episode: {target_episode_limit}")

        while episode < target_episode_limit:
            if not unseen_only:
                random.shuffle(train_routes)

            for selected_route in train_routes:
                if episode >= target_episode_limit:
                    break

                env = self.create_env(selected_route, use_gui=False)

                observation, info = env.reset()
                state = torch.tensor(observation, dtype=torch.float32, device=device)

                done = False
                truncated = False
                episode_reward = 0.0
                empty_counter = 0
                overtime_logged = False
                next_debug_step = float(self.num_seconds)

                total_queue_sum = 0.0
                total_waiting_time = 0.0
                total_co2_emission = 0.0
                vehicles_cleared = 0
                step_counter = 0
                prev_total_vehicles = None

                while not done:
                    action = self.select_action(
                        state_tensor=state,
                        observation=observation,
                        env=env,
                        epsilon=epsilon,
                        is_training=True
                    )

                    next_observation, reward, done, truncated, info = env.step(action)
                    next_state = torch.tensor(next_observation, dtype=torch.float32, device=device)

                    current_queue = float(sum(next_observation[13:21]))
                    total_queue_sum += current_queue

                    waiting_time = 0.0
                    for veh_id in env.sumo.vehicle.getIDList():
                        waiting_time += env.sumo.vehicle.getWaitingTime(veh_id)
                    total_waiting_time += waiting_time
                    total_co2_emission += self._get_step_total_co2_emission(env)

                    current_total_vehicles = env.sumo.simulation.getMinExpectedNumber()
                    if prev_total_vehicles is not None:
                        cleared = max(0, prev_total_vehicles - current_total_vehicles)
                        vehicles_cleared += cleared
                    prev_total_vehicles = current_total_vehicles

                    if env.sim_step >= self.num_seconds and current_total_vehicles > 0 and not overtime_logged:
                        overtime_logged = True
                        self.log(
                            f"Episode {episode + 1:03d} exceeded nominal duration {self.num_seconds}s "
                            f"and is continuing until traffic clears. Vehicles left: {current_total_vehicles}"
                        )

                    if env.sim_step >= self.num_seconds and current_total_vehicles > 0 and float(env.sim_step) >= next_debug_step:
                        self._log_overtime_debug_snapshot(env, next_observation, f"Episode {episode + 1:03d}")
                        next_debug_step += self.debug_interval_seconds

                    step_counter += 1

                    if self._traffic_is_finished(env):
                        empty_counter += 1
                    else:
                        empty_counter = 0

                    if truncated and current_total_vehicles > 0:
                        truncated = False

                    manual_terminal = empty_counter >= self.early_stop_patience
                    terminal_event = done or truncated or manual_terminal
                    if terminal_event:
                        reward = self._apply_terminal_remaining_penalty(reward, current_total_vehicles)

                    self.memory.append((state, action, next_state, reward, terminal_event))
                    self.optimize()

                    step_count += 1
                    if step_count % self.network_sync_rate == 0:
                        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())

                    state = next_state
                    observation = next_observation
                    episode_reward += reward

                    if manual_terminal:
                        self.log(
                            f"Episode {episode + 1:03d} early-stopped at sim step {env.sim_step} "
                            f"because traffic cleared."
                        )
                        break

                avg_queue = total_queue_sum / max(1, step_counter)
                remaining_vehicles = self._get_remaining_vehicles(env)
                overtime_used = env.sim_step > self.num_seconds
                cleared_all = remaining_vehicles == 0
                avg_co2_per_vehicle = total_co2_emission / max(1, vehicles_cleared)
                env.close()
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)

                rewards_per_episode.append(episode_reward)
                epsilon_history.append(epsilon)

                self.log(
                    f"Episode {episode + 1:03d} | "
                    f"Reward: {episode_reward:.2f} | "
                    f"Avg Queue: {avg_queue:.2f} | "
                    f"Waiting Time: {total_waiting_time:.2f}s | "
                    f"CO2: {total_co2_emission:.2f}mg | "
                    f"Vehicles Cleared: {vehicles_cleared} | "
                    f"Vehicles Left: {remaining_vehicles} | "
                    f"Overtime Used: {'Yes' if overtime_used else 'No'} | "
                    f"Cleared All: {'Yes' if cleared_all else 'No'} | "
                    f"Epsilon: {epsilon:.4f} | "
                    f"Route: {os.path.basename(selected_route)}"
                )

                if (episode + 1) % self.validation_interval == 0:
                    rl_validation = self.evaluate_controller_on_routes(self.val_routes, controller="rl")
                    validation_wait_history.append((episode + 1, rl_validation["total_waiting_time"]))

                    self.log("---------- VALIDATION CHECK ----------")
                    self.log(f"Episode                      : {episode + 1}")
                    self.log(f"RL Avg Queue                 : {rl_validation['avg_queue']:.4f}")
                    self.log(f"Fixed Avg Queue              : {fixed_baseline['avg_queue']:.4f}")
                    self.log(f"RL Total Waiting            : {rl_validation['total_waiting_time']:.4f}")
                    self.log(f"Fixed Total Waiting         : {fixed_baseline['total_waiting_time']:.4f}")
                    self.log(f"RL Total CO2                : {rl_validation['total_co2_emission']:.4f} mg")
                    self.log(f"Fixed Total CO2             : {fixed_baseline['total_co2_emission']:.4f} mg")
                    self.log(f"RL Vehicles Cleared         : {rl_validation['vehicles_cleared']:.4f}")
                    self.log(f"Fixed Vehicles Cleared      : {fixed_baseline['vehicles_cleared']:.4f}")
                    self.log(f"RL Vehicles Left            : {rl_validation['remaining_vehicles']:.4f}")
                    self.log(f"Fixed Vehicles Left         : {fixed_baseline['remaining_vehicles']:.4f}")
                    self.log(f"RL Avg Wait/Vehicle         : {rl_validation['avg_wait_per_vehicle']:.4f}")
                    self.log(f"Fixed Avg Wait/Vehicle      : {fixed_baseline['avg_wait_per_vehicle']:.4f}")
                    self.log(f"RL Avg CO2/Vehicle          : {rl_validation['avg_co2_per_vehicle']:.4f} mg")
                    self.log(f"Fixed Avg CO2/Vehicle       : {fixed_baseline['avg_co2_per_vehicle']:.4f} mg")
                    self.log("--------------------------------------")

                    if rl_validation["total_waiting_time"] < best_validation_wait:
                        best_validation_wait = rl_validation["total_waiting_time"]
                        self._save_checkpoint(
                            checkpoint_path=self.best_model_file,
                            episode=episode + 1,
                            epsilon=epsilon,
                            step_count=step_count,
                            rewards_per_episode=rewards_per_episode,
                            epsilon_history=epsilon_history,
                            validation_wait_history=validation_wait_history,
                            best_validation_wait=best_validation_wait,
                            fixed_baseline=fixed_baseline,
                        )
                        self.log(
                            f"New best model saved to {self.best_model_file} based on validation waiting time = "
                            f"{best_validation_wait:.4f}"
                        )
                    else:
                        self.log(
                            f"No new best model at episode {episode + 1}. "
                            f"Current best validation waiting time remains {best_validation_wait:.4f} "
                            f"at {self.best_model_file}"
                        )

                if (episode + 1) % 10 == 0:
                    self.save_graph(rewards_per_episode, epsilon_history, validation_wait_history)
                    self._save_checkpoint(
                        checkpoint_path=self.final_model_file,
                        episode=episode + 1,
                        epsilon=epsilon,
                        step_count=step_count,
                        rewards_per_episode=rewards_per_episode,
                        epsilon_history=epsilon_history,
                        validation_wait_history=validation_wait_history,
                        best_validation_wait=best_validation_wait,
                        fixed_baseline=fixed_baseline,
                    )

                episode += 1

        if not os.path.exists(self.best_model_file):
            self._save_checkpoint(
                checkpoint_path=self.best_model_file,
                episode=episode,
                epsilon=epsilon,
                step_count=step_count,
                rewards_per_episode=rewards_per_episode,
                epsilon_history=epsilon_history,
                validation_wait_history=validation_wait_history,
                best_validation_wait=best_validation_wait,
                fixed_baseline=fixed_baseline,
            )
            self.log(f"No validation checkpoint improved. Best model snapshot saved to {self.best_model_file}")

        self._save_checkpoint(
            checkpoint_path=self.final_model_file,
            episode=episode,
            epsilon=epsilon,
            step_count=step_count,
            rewards_per_episode=rewards_per_episode,
            epsilon_history=epsilon_history,
            validation_wait_history=validation_wait_history,
            best_validation_wait=best_validation_wait,
            fixed_baseline=fixed_baseline,
        )
        self.log(f"Final model saved to {self.final_model_file}")

        self.save_graph(rewards_per_episode, epsilon_history, validation_wait_history)
        self.log("Training finished.")

    def test(self, route_name=None, list_routes=False, model_path=None):
        if list_routes:
            test_routes = self._get_test_routes()
            if len(test_routes) == 0:
                self.log("No test routes found.")
                return
            self.log("Available test routes:")
            for route_path in test_routes:
                self.log(f"- {os.path.basename(route_path)}")
            return

        test_routes = self._get_test_routes()
        if len(test_routes) == 0:
            self.log("No test routes found.")
            return

        if route_name is not None:
            matching_routes = [
                route_path for route_path in test_routes if os.path.basename(route_path) == route_name
            ]
            if len(matching_routes) == 0:
                route_root = Path(self.route_dir).parent
                all_matching_routes = [
                    str(route_path)
                    for route_path in route_root.rglob(route_name)
                    if route_path.is_file()
                ]
                if len(all_matching_routes) == 0:
                    available_routes = ", ".join(os.path.basename(path) for path in test_routes)
                    raise ValueError(f"Route '{route_name}' not found in test routes or route directory. Available test routes: {available_routes}")
                selected_route = all_matching_routes[0]
            else:
                selected_route = matching_routes[0]
        else:
            selected_route = random.choice(test_routes)
        self.log(f"Selected test route: {os.path.basename(selected_route)}")

        env = self.create_env(selected_route, use_gui=True)

        observation, info = env.reset()
        state_dim = len(observation)
        action_dim = env.action_space.n

        self.policy_dqn = SumoDQN(state_dim, action_dim, self.fc1_nodes).to(device)
        model_to_load = (
            self._resolve_resume_model_file(model_path)
            if model_path is not None
            else self._get_latest_model_file()
        )
        self.log(f"Loading model file: {model_to_load}")
        self.policy_dqn.load_state_dict(self._load_policy_for_inference(model_to_load))
        self.policy_dqn.eval()

        done = False
        truncated = False
        total_reward = 0.0
        empty_counter = 0
        overtime_logged = False

        total_queue_normalized_sum = 0.0
        total_queue_raw_sum = 0.0
        total_waiting_time = 0.0
        total_co2_emission = 0.0
        vehicles_cleared = 0
        step_counter = 0
        prev_total_vehicles = None
        best_remaining_vehicles = float("inf")
        last_progress_sim_step = 0.0
        directional_queue_rows = []

        while not done:
            state = torch.tensor(observation, dtype=torch.float32, device=device)

            action = self.select_action(
                state_tensor=state,
                observation=observation,
                env=env,
                epsilon=0.0,
                is_training=False
            )

            next_observation, reward, done, truncated, info = env.step(action)
            actual_phase_after_step = get_current_phase_from_obs(next_observation)

            current_queue_normalized = float(sum(next_observation[13:21]))
            total_queue_normalized_sum += current_queue_normalized

            traffic_signal = env.traffic_signals[env.ts_ids[0]]
            current_queue_raw = float(traffic_signal.get_total_queued())
            total_queue_raw_sum += current_queue_raw
            ns_queue_raw, we_queue_raw = self._get_directional_queue_raw(env)
            directional_queue_rows.append((step_counter + 1, float(env.sim_step), ns_queue_raw, we_queue_raw))

            waiting_time = 0.0
            for veh_id in env.sumo.vehicle.getIDList():
                waiting_time += env.sumo.vehicle.getWaitingTime(veh_id)
            total_waiting_time += waiting_time
            total_co2_emission += self._get_step_total_co2_emission(env)

            current_total_vehicles = env.sumo.simulation.getMinExpectedNumber()
            if prev_total_vehicles is not None:
                vehicles_cleared += max(0, prev_total_vehicles - current_total_vehicles)
            prev_total_vehicles = current_total_vehicles

            if current_total_vehicles < best_remaining_vehicles:
                best_remaining_vehicles = current_total_vehicles
                last_progress_sim_step = float(env.sim_step)

            if env.sim_step >= self.num_seconds and current_total_vehicles > 0 and not overtime_logged:
                overtime_logged = True
                self.log(
                    f"Test exceeded nominal duration {self.num_seconds}s and is continuing until "
                    f"clearance or hard cap. Vehicles left: {current_total_vehicles}"
                )

            step_counter += 1

            if truncated and current_total_vehicles > 0:
                truncated = False

            stalled_terminal = (
                current_total_vehicles > 0
                and (float(env.sim_step) - last_progress_sim_step) >= self.no_progress_patience_seconds
            )
            if stalled_terminal:
                self.log(
                    f"Test stopped after {self.no_progress_patience_seconds}s without clearance progress. "
                    f"Vehicles left: {current_total_vehicles}"
                )

            terminal_event = done or truncated or empty_counter >= self.early_stop_patience or stalled_terminal
            if terminal_event:
                reward = self._apply_terminal_remaining_penalty(reward, current_total_vehicles)

            total_reward += reward

            self.log(f"\n[Step {env.sim_step}]")
            self.log(f"Executed action         : {action} -> {PHASE_MAP.get(action, 'Unknown Phase')}")
            self.log(f"Actual phase after step : {actual_phase_after_step} -> {PHASE_MAP.get(actual_phase_after_step, 'Unknown Phase')}")
            self.log(f"Phase one-hot           : {next_observation[:4]}")
            self.log(f"Min green               : {next_observation[4]}")
            self.log(f"Queues normalized       : {next_observation[13:21]}")
            self.log(f"Queue raw total         : {current_queue_raw:.0f}")
            self.log(f"Reward                  : {reward:.8f}")
            self.log(f"Info                    : {info}")

            if self._traffic_is_finished(env):
                empty_counter += 1
            else:
                empty_counter = 0

            if empty_counter >= self.early_stop_patience or stalled_terminal:
                self.log(
                    f"Test early-stopped at sim step {env.sim_step} because "
                    f"{'traffic cleared' if empty_counter >= self.early_stop_patience else 'clearance progress stalled'}."
                )
                break

            observation = next_observation

        avg_queue_normalized = total_queue_normalized_sum / max(1, step_counter)
        avg_queue_raw = total_queue_raw_sum / max(1, step_counter)
        avg_wait_per_vehicle = total_waiting_time / max(1, vehicles_cleared)
        avg_co2_per_vehicle = total_co2_emission / max(1, vehicles_cleared)
        avg_ns_queue_raw = sum(row[2] for row in directional_queue_rows) / max(1, len(directional_queue_rows))
        avg_we_queue_raw = sum(row[3] for row in directional_queue_rows) / max(1, len(directional_queue_rows))
        remaining_vehicles = self._get_remaining_vehicles(env)
        overtime_used = env.sim_step > self.num_seconds
        cleared_all = remaining_vehicles == 0
        env.close()

        queue_csv_path = os.path.splitext(self.log_file)[0] + "_directional_queue.csv"
        queue_plot_path = os.path.splitext(self.log_file)[0] + "_directional_queue.png"
        self._write_directional_queue_outputs(
            queue_csv_path,
            queue_plot_path,
            directional_queue_rows,
            "Histogram and distribution of queue lengths for DRL agent",
        )

        self.log("========== TEST SUMMARY ==========")
        self.log(f"Route file           : {os.path.basename(selected_route)}")
        self.log(f"Total reward         : {total_reward:.8f}")
        self.log(f"Average queue norm   : {avg_queue_normalized:.4f}")
        self.log(f"Average queue raw    : {avg_queue_raw:.4f}")
        self.log(f"Average NS queue raw : {avg_ns_queue_raw:.4f}")
        self.log(f"Average WE queue raw : {avg_we_queue_raw:.4f}")
        self.log(f"Total waiting time   : {total_waiting_time:.4f}")
        self.log(f"Total CO2 emission   : {total_co2_emission:.4f} mg")
        self.log(f"Vehicles cleared     : {vehicles_cleared}")
        self.log(f"Vehicles left        : {remaining_vehicles}")
        self.log(f"Overtime used        : {'Yes' if overtime_used else 'No'}")
        self.log(f"Cleared all vehicles : {'Yes' if cleared_all else 'No'}")
        self.log(f"Average wait/vehicle : {avg_wait_per_vehicle:.4f}")
        self.log(f"Average CO2/vehicle  : {avg_co2_per_vehicle:.4f} mg")
        self.log(f"Simulation steps     : {step_counter}")
        self.log(f"Directional queue CSV: {queue_csv_path}")
        self.log(f"Directional queue PNG: {queue_plot_path}")
        self.log("==================================")

    def _get_latest_model_file(self):
        model_files = sorted(
            [
                os.path.join(self.runs_dir, file_name)
                for file_name in os.listdir(self.runs_dir)
                if file_name.startswith("sumo1_per_lane_") and file_name.endswith(".pt")
            ]
        )

        if len(model_files) == 0:
            raise FileNotFoundError(f"No saved per-lane model files found in: {self.runs_dir}")

        return model_files[-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SUMO DQN Agent With Per-Lane Phase Features")
    parser.add_argument("hyperparameters", help="Name of hyperparameter config in YAML file")
    parser.add_argument("--train", action="store_true", help="Train the DQN agent")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        default=None,
        help="Resume training from a checkpoint path or the latest checkpoint if no path is provided",
    )
    parser.add_argument(
        "--resume-unseen",
        action="store_true",
        help="When resuming, continue training only on training routes not yet seen in the run log",
    )
    parser.add_argument(
        "--target-episodes",
        type=int,
        default=None,
        help="Optional total episode target for resumed training, for example 840",
    )
    parser.add_argument(
        "--route",
        default=None,
        help="Specific test route filename to run, for example _106_18_17_104.rou.xml",
    )
    parser.add_argument(
        "--list-test-routes",
        action="store_true",
        help="List the route filenames in the test split and exit",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Specific checkpoint file to load for testing",
    )
    args = parser.parse_args()

    hyperparameters = load_hyperparameters(args.hyperparameters)
    agent = Agent(hyperparameters)

    if args.train:
        agent.train(
            resume_from=args.resume,
            unseen_only=args.resume_unseen,
            target_episodes=args.target_episodes,
        )
    else:
        agent.test(
            route_name=args.route,
            list_routes=args.list_test_routes,
            model_path=args.model,
        )
