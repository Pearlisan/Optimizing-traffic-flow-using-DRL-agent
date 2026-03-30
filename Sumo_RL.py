# ============================================================
# BLOCK 1: Import libraries
# ============================================================
import argparse
import os
import random
import re
import time
from datetime import datetime
from pathlib import Path

import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from sumo_rl import SumoEnvironment

from dqn_sumo import SumoDQN
from experience_replay_sumo import ReplayMemory


# ============================================================
# BLOCK 2: Set device
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
SCRIPT_DIR = Path(__file__).resolve().parent
REWARD_OVERTIME_STEP_PENALTY = 0.0


# ============================================================
# BLOCK 3: Map action indices to readable phase names
# ============================================================
PHASE_MAP = {
    0: "Phase 1: North-South Through",
    1: "Phase 2: East-West Through",
    2: "Phase 3: North-South Protected Left",
    3: "Phase 4: East-West Protected Left"
}


# ============================================================
# BLOCK 4: Helper function to load hyperparameters from YAML
# ============================================================
def load_hyperparameters(config_name, yaml_file="hyperparameters_crossnetwork.yml"):
    yaml_path = Path(yaml_file)
    if not yaml_path.is_absolute():
        yaml_path = SCRIPT_DIR / yaml_path

    with open(yaml_path, "r") as file:
        configs = yaml.safe_load(file)

    if config_name not in configs:
        raise ValueError(f"Config '{config_name}' not found in {yaml_path}")

    return configs[config_name]


# ============================================================
# BLOCK 5: Helper function to get all route files
# ============================================================
def get_route_files(route_dir):
    route_files = [
        os.path.join(route_dir, f)
        for f in os.listdir(route_dir)
        if f.endswith(".rou.xml")
    ]

    if len(route_files) == 0:
        raise FileNotFoundError(f"No .rou.xml files found in: {route_dir}")

    return route_files


# ============================================================
# BLOCK 6: Helper function to decode phase from observation
# ============================================================
def get_current_phase_from_obs(observation):
    """
    Extract current green phase index from one-hot encoding.
    Assumes first 4 positions are phase one-hot values.
    """
    phase_one_hot = observation[:4]
    return int(np.argmax(phase_one_hot))


# ============================================================
# BLOCK 7: Better reward function based on queue reduction
# ============================================================
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

    reward = (
        -0.001 * total_wait
        -0.1 * current_queue
        + 0.5 * cleared
        -0.05 * current_total_vehicles
    )

    nominal_seconds = getattr(ts.env, "_nominal_num_seconds", None)
    if (
        nominal_seconds is not None
        and getattr(ts.env, "sim_step", 0) >= nominal_seconds
        and current_total_vehicles > 0
    ):
        reward -= REWARD_OVERTIME_STEP_PENALTY

    return float(reward)

# ============================================================
# BLOCK 8: Define agent class
# ============================================================
class Agent:
    def __init__(self, hyperparameters):
        self.hp = hyperparameters

        # Set random seeds
        random.seed(self.hp["random_seed"])
        np.random.seed(self.hp["random_seed"])
        torch.manual_seed(self.hp["random_seed"])

        # Environment file paths
        self.net_file = self._resolve_path(self.hp["net_file"])
        self.route_dir = self._resolve_path(self.hp["route_dir"])

        # DQN hyperparameters
        self.replay_memory_size = self.hp["replay_memory_size"]
        self.mini_batch_size = self.hp["mini_batch_size"]
        self.epsilon_init = self.hp["epsilon_init"]
        self.epsilon_decay = self.hp["epsilon_decay"]
        self.epsilon_min = self.hp["epsilon_min"]
        self.network_sync_rate = self.hp["network_sync_rate"]
        self.learning_rate_a = self.hp["learning_rate_a"]
        self.discount_factor_g = self.hp["discount_factor_g"]
        self.fc1_nodes = self.hp["fc1_nodes"]
        self.num_episodes = self.hp["num_episodes"]

        # SUMO hyperparameters
        self.num_seconds = self.hp["num_seconds"]
        self.delta_time = self.hp["delta_time"]
        self.yellow_time = self.hp["yellow_time"]
        self.min_green = self.hp["min_green"]
        self.use_gui = self.hp["use_gui"]
        self.sumo_warnings = self.hp["sumo_warnings"]
        self.additional_sumo_cmd = self.hp["additional_sumo_cmd"]
        self.clearance_extra_seconds = self.hp.get("clearance_extra_seconds", 600)
        self.terminal_remaining_penalty = self.hp.get("terminal_remaining_penalty", 10.0)
        self.overtime_step_penalty = self.hp.get("overtime_step_penalty", 0.0)
        self.no_progress_patience_seconds = self.hp.get("no_progress_patience_seconds", 600)

        # Early-stop hyperparameters
        self.early_stop_patience = self.hp.get("early_stop_patience", 3)

        # Validation settings
        self.validation_interval = self.hp.get("validation_interval", 25)
        self.validation_routes_count = self.hp.get("validation_routes_count", 5)

        # Runtime paths
        self.runs_dir = str(SCRIPT_DIR / "runs")
        os.makedirs(self.runs_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.timestamp = timestamp
        self._set_run_artifact_paths(os.path.join(self.runs_dir, f"sumo1_{timestamp}.pt"))

        # Loss function
        self.loss_fn = nn.MSELoss()

    def _resolve_path(self, path_value):
        path = Path(path_value)
        if not path.is_absolute():
            path = SCRIPT_DIR / path
        return str(path)

    def _set_run_artifact_paths(self, model_file):
        model_path = Path(model_file)
        if not model_path.is_absolute():
            model_path = Path(self.runs_dir) / model_path

        stem = model_path.stem
        if stem.endswith("_best"):
            run_stem = stem[:-5]
        elif stem.endswith("_final"):
            run_stem = stem[:-6]
        else:
            run_stem = stem

        run_path = model_path.with_name(run_stem)

        self.model_file = str(model_path)
        self.best_model_file = str(run_path.with_name(f"{run_stem}_best.pt"))
        self.final_model_file = str(run_path.with_name(f"{run_stem}_final.pt"))
        self.log_file = str(run_path.with_suffix(".log"))
        self.graph_file = str(run_path.with_suffix(".png"))

        if run_stem.startswith("sumo1_"):
            self.timestamp = run_stem[len("sumo1_"):]

    def log(self, message):
        """
        Print to console and save to log file.
        """
        print(message)
        with open(self.log_file, "a", encoding="utf-8") as file:
            file.write(message + "\n")

    def create_env(self, route_file, use_gui=False):
        """
        Create SUMO environment using the queue-improvement reward.
        """
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
            sumo_warnings=self.sumo_warnings,
            additional_sumo_cmd=self.additional_sumo_cmd if use_gui else ""
        )
        env._nominal_num_seconds = self.num_seconds
        return env

    def _extract_seen_training_routes_from_log(self, log_file):
        if not os.path.exists(log_file):
            return set()

        route_pattern = re.compile(r"Episode\s+\d+\s+\|.*?\|\s+Route:\s+(.+?\.rou\.xml)")
        seen_routes = set()

        with open(log_file, "r", encoding="utf-8") as file:
            for line in file:
                match = route_pattern.search(line)
                if match:
                    seen_routes.add(match.group(1).strip())

        return seen_routes

    def select_action(self, state_tensor, observation, env, epsilon, is_training=True):
        """
        Select action using epsilon-greedy policy while respecting min_green.
        """
        current_phase = get_current_phase_from_obs(observation)
        min_green_flag = int(observation[4])

        # Force current phase if min green not yet satisfied
        if min_green_flag == 0:
            return current_phase

        # Exploration
        if is_training and random.random() < epsilon:
            return env.action_space.sample()

        # Exploitation
        with torch.no_grad():
            q_values = self.policy_dqn(state_tensor.unsqueeze(0)).squeeze(0)
            return int(torch.argmax(q_values).item())

    def select_fixed_action(self, observation, fixed_state):
        """
        Simple cyclic fixed-time style controller.

        It keeps the current phase until min_green is satisfied.
        Once min_green is satisfied, it moves to the next phase in order:
        0 -> 1 -> 2 -> 3 -> 0 -> ...
        """
        current_phase = get_current_phase_from_obs(observation)
        min_green_flag = int(observation[4])

        if min_green_flag == 0:
            return current_phase

        if fixed_state["last_phase"] is None:
            fixed_state["last_phase"] = current_phase
            return current_phase

        next_phase = (fixed_state["last_phase"] + 1) % 4
        fixed_state["last_phase"] = next_phase
        return next_phase

    def optimize(self):
        """
        Sample a mini-batch from replay memory and update policy network.
        """
        if len(self.memory) < self.mini_batch_size:
            return

        mini_batch = self.memory.sample(self.mini_batch_size)
        states, actions, next_states, rewards, dones = zip(*mini_batch)

        states = torch.stack(states).to(device)
        actions = torch.tensor(actions, dtype=torch.int64, device=device)
        next_states = torch.stack(next_states).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        dones = torch.tensor(dones, dtype=torch.float32, device=device)

        current_q = self.policy_dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            max_next_q = self.target_dqn(next_states).max(dim=1)[0]
            target_q = rewards + (1 - dones) * self.discount_factor_g * max_next_q

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_graph(self, rewards_per_episode, epsilon_history, validation_wait_history):
        """
        Save reward curve plot, epsilon decay plot, and validation waiting time.
        """
        nrows = 3 if len(validation_wait_history) > 0 else 2
        fig, axes = plt.subplots(nrows, 1, figsize=(10, 4 * nrows), sharex=True)

        if nrows == 2:
            ax1, ax2 = axes
            ax3 = None
        else:
            ax1, ax2, ax3 = axes

        rewards_array = np.array(rewards_per_episode)

        ax1.plot(rewards_array, label="Episode Reward")
        ax1.set_ylabel("Reward")
        ax1.set_title("Episode Reward")
        ax1.grid(True)
        ax1.legend(loc="best")

        ax2.plot(epsilon_history, linestyle="--", label="Epsilon")
        ax2.set_ylabel("Epsilon")
        ax2.set_title("Epsilon Decay")
        ax2.grid(True)
        ax2.legend(loc="best")

        if ax3 is not None:
            val_episodes = [item[0] for item in validation_wait_history]
            val_waits = [item[1] for item in validation_wait_history]
            ax3.plot(val_episodes, val_waits, marker="o", linestyle="-.", label="Validation Total Waiting")
            ax3.set_ylabel("Total Waiting")
            ax3.set_title("Validation Total Waiting")
            ax3.grid(True)
            ax3.legend(loc="best")

        axes[-1].set_xlabel("Episode")

        fig.tight_layout()
        plt.savefig(self.graph_file)
        plt.close()

    def _traffic_is_finished(self, env):
        """
        Return True only when the simulation is actually empty.
        """
        try:
            return env.sumo.simulation.getMinExpectedNumber() == 0
        except Exception:
            return False

    def _get_remaining_vehicles(self, env):
        """
        Return how many vehicles SUMO still expects in the network.
        """
        try:
            return int(env.sumo.simulation.getMinExpectedNumber())
        except Exception:
            return -1

    def _get_latest_model_file(self):
        """
        Return the newest timestamped model file in the runs directory.
        """
        model_files = sorted(
            [
                os.path.join(self.runs_dir, file_name)
                for file_name in os.listdir(self.runs_dir)
                if file_name.startswith("sumo1_") and file_name.endswith(".pt")
            ]
        )

        if len(model_files) == 0:
            raise FileNotFoundError(f"No saved model files found in: {self.runs_dir}")

        return model_files[-1]

    def _resolve_resume_model_file(self, resume_from):
        if resume_from in (None, "", "latest"):
            return self._get_latest_model_file()

        resume_path = Path(resume_from)
        if not resume_path.is_absolute():
            resume_path = SCRIPT_DIR / resume_path

        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

        return str(resume_path)

    def _resolve_test_model_file(self, model_path):
        if model_path in (None, "", "latest"):
            return self._get_latest_model_file()

        resolved_path = Path(model_path)
        if not resolved_path.is_absolute():
            resolved_path = SCRIPT_DIR / resolved_path

        if not resolved_path.exists():
            raise FileNotFoundError(f"Test checkpoint not found: {resolved_path}")

        return str(resolved_path)

    def _extract_training_state_from_log(self, log_file):
        """
        Recover lightweight training progress from an old text log.

        This is used as a fallback for legacy checkpoints that only stored
        model weights and not the optimizer/replay state.
        """
        if not os.path.exists(log_file):
            return None

        episode_pattern = re.compile(
            r"Episode\s+(\d+)\s+\|\s+Reward:\s+([-0-9.]+).*?\|\s+Epsilon:\s+([0-9.]+)\s+\|"
        )
        validation_episode_pattern = re.compile(r"Episode\s*:\s*(\d+)")
        validation_wait_pattern = re.compile(r"RL Total Waiting\s*:\s*([0-9.]+)")
        best_validation_pattern = re.compile(r"validation waiting time =\s*([0-9.]+)")

        rewards_per_episode = []
        epsilon_history = []
        validation_wait_history = []
        best_validation_wait = float("inf")
        pending_validation_episode = None

        with open(log_file, "r", encoding="utf-8") as file:
            for line in file:
                episode_match = episode_pattern.search(line)
                if episode_match:
                    rewards_per_episode.append(float(episode_match.group(2)))
                    epsilon_history.append(float(episode_match.group(3)))
                    continue

                validation_episode_match = validation_episode_pattern.search(line)
                if validation_episode_match:
                    pending_validation_episode = int(validation_episode_match.group(1))
                    continue

                validation_wait_match = validation_wait_pattern.search(line)
                if validation_wait_match and pending_validation_episode is not None:
                    validation_wait_history.append(
                        (pending_validation_episode, float(validation_wait_match.group(1)))
                    )
                    pending_validation_episode = None
                    continue

                best_validation_match = best_validation_pattern.search(line)
                if best_validation_match:
                    best_validation_wait = min(
                        best_validation_wait,
                        float(best_validation_match.group(1))
                    )

        if len(rewards_per_episode) == 0:
            return None

        return {
            "episode": len(rewards_per_episode),
            "epsilon": epsilon_history[-1] if epsilon_history else self.epsilon_init,
            "step_count": 0,
            "rewards_per_episode": rewards_per_episode,
            "epsilon_history": epsilon_history,
            "validation_wait_history": validation_wait_history,
            "best_validation_wait": best_validation_wait,
            "memory": None,
            "optimizer_state_dict": None,
            "target_state_dict": None,
            "fixed_baseline": None,
        }

    def _save_checkpoint(
        self,
        checkpoint_path,
        episode,
        epsilon,
        step_count,
        rewards_per_episode,
        epsilon_history,
        validation_wait_history,
        best_validation_wait,
        fixed_baseline,
    ):
        checkpoint = {
            "checkpoint_version": 2,
            "policy_state_dict": self.policy_dqn.state_dict(),
            "target_state_dict": self.target_dqn.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "episode": episode,
            "epsilon": epsilon,
            "step_count": step_count,
            "rewards_per_episode": rewards_per_episode,
            "epsilon_history": epsilon_history,
            "validation_wait_history": validation_wait_history,
            "best_validation_wait": best_validation_wait,
            "memory": list(self.memory.memory),
            "fixed_baseline": fixed_baseline,
        }
        torch.save(checkpoint, checkpoint_path)

    def _load_checkpoint_data(self, model_path):
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        if isinstance(checkpoint, dict) and "policy_state_dict" in checkpoint:
            return checkpoint

        if isinstance(checkpoint, dict):
            log_fallback = self._extract_training_state_from_log(
                str(Path(model_path).with_suffix(".log"))
            )
            if log_fallback is None:
                log_fallback = {
                    "episode": 0,
                    "epsilon": self.epsilon_init,
                    "step_count": 0,
                    "rewards_per_episode": [],
                    "epsilon_history": [],
                    "validation_wait_history": [],
                    "best_validation_wait": float("inf"),
                    "memory": None,
                    "optimizer_state_dict": None,
                    "target_state_dict": None,
                    "fixed_baseline": None,
                }

            log_fallback["policy_state_dict"] = checkpoint
            return log_fallback

        raise ValueError(f"Unsupported checkpoint format in {model_path}")

    def _load_policy_for_inference(self, model_path):
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "policy_state_dict" in checkpoint:
            return checkpoint["policy_state_dict"]
        return checkpoint

    def _apply_terminal_remaining_penalty(self, reward, remaining_vehicles):
        """
        Penalize terminal states that still leave vehicles in the network.
        """
        if remaining_vehicles > 0:
            reward -= self.terminal_remaining_penalty * remaining_vehicles
        return reward

    def run_single_episode(self, env, controller="rl"):
        """
        Run a single episode using either the RL controller or the fixed controller.
        Returns benchmark metrics for that episode.
        """
        observation, info = env.reset()
        done = False
        truncated = False
        total_reward = 0.0
        empty_counter = 0
        overtime_logged = False

        total_queue_normalized_sum = 0.0
        total_queue_raw_sum = 0.0
        total_waiting_time = 0.0
        vehicles_cleared = 0
        step_counter = 0
        prev_total_vehicles = None
        best_remaining_vehicles = float("inf")
        last_progress_sim_step = 0.0

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
                    f"Episode exceeded nominal duration {self.num_seconds}s and is continuing "
                    f"until clearance or hard cap. Vehicles left: {current_total_vehicles}"
                )

            step_counter += 1

            if self._traffic_is_finished(env):
                empty_counter += 1
            else:
                empty_counter = 0

            if truncated and current_total_vehicles > 0:
                truncated = False

            stalled_terminal = (
                current_total_vehicles > 0
                and (float(env.sim_step) - last_progress_sim_step) >= self.no_progress_patience_seconds
            )
            if stalled_terminal:
                self.log(
                    f"Episode stopped after {self.no_progress_patience_seconds}s without clearance progress. "
                    f"Vehicles left: {current_total_vehicles}"
                )

            terminal_event = done or truncated or empty_counter >= self.early_stop_patience or stalled_terminal
            if terminal_event:
                reward = self._apply_terminal_remaining_penalty(reward, current_total_vehicles)

            total_reward += reward

            if empty_counter >= self.early_stop_patience or stalled_terminal:
                break

            observation = next_observation

        avg_queue = total_queue_normalized_sum / max(1, step_counter)
        avg_wait_per_vehicle = total_waiting_time / max(1, vehicles_cleared)
        remaining_vehicles = self._get_remaining_vehicles(env)

        return {
            "total_reward": total_reward,
            "avg_queue": avg_queue,
            "total_waiting_time": total_waiting_time,
            "vehicles_cleared": vehicles_cleared,
            "avg_wait_per_vehicle": avg_wait_per_vehicle,
            "simulation_steps": step_counter,
            "remaining_vehicles": remaining_vehicles,
        }

    def evaluate_controller_on_routes(self, routes, controller="rl"):
        """
        Evaluate either the RL controller or the fixed controller on a list of routes.
        Returns averaged metrics.
        """
        metrics = []

        for route_file in routes:
            env = self.create_env(route_file, use_gui=False)
            result = self.run_single_episode(env, controller=controller)
            env.close()
            metrics.append(result)

        summary = {}
        keys = metrics[0].keys()
        for key in keys:
            summary[key] = float(np.mean([m[key] for m in metrics]))

        return summary

    def train(self, resume_from=None, unseen_only=False):
        """
        Train DQN agent and validate periodically against a fixed controller baseline.
        """
        fixed_baseline = None
        resume_checkpoint = None
        global REWARD_OVERTIME_STEP_PENALTY
        REWARD_OVERTIME_STEP_PENALTY = self.overtime_step_penalty

        if resume_from is not None:
            resume_model_file = self._resolve_resume_model_file(resume_from)
            self._set_run_artifact_paths(resume_model_file)
            resume_checkpoint = self._load_checkpoint_data(resume_model_file)
            self.log(f"Resuming training from checkpoint: {resume_model_file}")

        self.log("Starting training...")
        self.log(f"Best checkpoint path : {self.best_model_file}")
        self.log(f"Final checkpoint path: {self.final_model_file}")
        self.log(f"Run log path         : {self.log_file}")
        self.log(f"Run plot path        : {self.graph_file}")

        # get all route files and create a reproducible split
        all_routes = sorted(get_route_files(self.route_dir))
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
            seen_route_names = self._extract_seen_training_routes_from_log(self.log_file)
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

        # temporary environment to get state/action dimensions
        temp_route = train_routes[0]
        temp_env = self.create_env(temp_route, use_gui=False)
        initial_observation, _ = temp_env.reset()

        state_dim = len(initial_observation)
        action_dim = temp_env.action_space.n
        temp_env.close()

        self.log(f"State dimension: {state_dim}")
        self.log(f"Action dimension: {action_dim}")

        # create policy and target networks
        self.policy_dqn = SumoDQN(state_dim, action_dim, self.fc1_nodes).to(device)
        self.target_dqn = SumoDQN(state_dim, action_dim, self.fc1_nodes).to(device)
        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())

        # optimizer and replay memory
        self.optimizer = torch.optim.Adam(self.policy_dqn.parameters(), lr=self.learning_rate_a)
        self.memory = ReplayMemory(self.replay_memory_size)

        epsilon = self.epsilon_init
        step_count = 0
        rewards_per_episode = []
        epsilon_history = []
        validation_wait_history = []
        best_validation_wait = float("inf")
        episode = 0

        if resume_checkpoint is not None:
            self.policy_dqn.load_state_dict(resume_checkpoint["policy_state_dict"])

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

            self.log(
                f"Recovered progress up to episode {episode} with epsilon {epsilon:.4f}."
            )
            if resume_checkpoint.get("optimizer_state_dict") is None:
                self.log(
                    "Legacy checkpoint detected: model weights were restored, "
                    "but optimizer and replay memory were not available."
                )

        # establish one fixed baseline before training
        if fixed_baseline is None:
            fixed_baseline = self.evaluate_controller_on_routes(self.val_routes, controller="fixed")

        self.log("========== FIXED CONTROLLER VALIDATION BASELINE ==========")
        self.log(f"Validation Avg Queue        : {fixed_baseline['avg_queue']:.4f}")
        self.log(f"Validation Total Waiting   : {fixed_baseline['total_waiting_time']:.4f}")
        self.log(f"Validation Vehicles Cleared: {fixed_baseline['vehicles_cleared']:.4f}")
        self.log(f"Validation Avg Wait/Vehicle: {fixed_baseline['avg_wait_per_vehicle']:.4f}")
        self.log("=========================================================")

        target_episode_limit = self.num_episodes
        if unseen_only:
            target_episode_limit = min(self.num_episodes, episode + len(train_routes))
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

                total_queue_sum = 0.0
                total_waiting_time = 0.0
                vehicles_cleared = 0
                step_counter = 0
                prev_total_vehicles = None
                best_remaining_vehicles = float("inf")
                last_progress_sim_step = 0.0

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

                    current_total_vehicles = env.sumo.simulation.getMinExpectedNumber()
                    if prev_total_vehicles is not None:
                        cleared = max(0, prev_total_vehicles - current_total_vehicles)
                        vehicles_cleared += cleared
                    prev_total_vehicles = current_total_vehicles

                    if current_total_vehicles < best_remaining_vehicles:
                        best_remaining_vehicles = current_total_vehicles
                        last_progress_sim_step = float(env.sim_step)

                    if env.sim_step >= self.num_seconds and current_total_vehicles > 0 and not overtime_logged:
                        overtime_logged = True
                        self.log(
                            f"Episode {episode + 1:03d} exceeded nominal duration {self.num_seconds}s "
                            f"and is continuing until clearance or hard cap. Vehicles left: "
                            f"{current_total_vehicles}"
                        )

                    step_counter += 1

                    if self._traffic_is_finished(env):
                        empty_counter += 1
                    else:
                        empty_counter = 0

                    if truncated and current_total_vehicles > 0:
                        truncated = False

                    manual_terminal = empty_counter >= self.early_stop_patience
                    stalled_terminal = (
                        current_total_vehicles > 0
                        and (float(env.sim_step) - last_progress_sim_step) >= self.no_progress_patience_seconds
                    )
                    if stalled_terminal:
                        self.log(
                            f"Episode {episode + 1:03d} stopped after "
                            f"{self.no_progress_patience_seconds}s without clearance progress. "
                            f"Vehicles left: {current_total_vehicles}"
                        )

                    terminal_event = done or truncated or manual_terminal or stalled_terminal
                    if terminal_event:
                        reward = self._apply_terminal_remaining_penalty(reward, current_total_vehicles)

                    self.memory.append(
                        (state, action, next_state, reward, terminal_event)
                    )

                    self.optimize()

                    step_count += 1
                    if step_count % self.network_sync_rate == 0:
                        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())

                    state = next_state
                    observation = next_observation
                    episode_reward += reward

                    if manual_terminal or stalled_terminal:
                        self.log(
                            f"Episode {episode + 1:03d} early-stopped at sim step {env.sim_step} "
                            f"because {'traffic cleared' if manual_terminal else 'clearance progress stalled'}."
                        )
                        break

                avg_queue = total_queue_sum / max(1, step_counter)
                remaining_vehicles = self._get_remaining_vehicles(env)
                overtime_used = env.sim_step > self.num_seconds
                cleared_all = remaining_vehicles == 0
                env.close()
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)

                rewards_per_episode.append(episode_reward)
                epsilon_history.append(epsilon)

                self.log(
                    f"Episode {episode + 1:03d} | "
                    f"Reward: {episode_reward:.2f} | "
                    f"Avg Queue: {avg_queue:.2f} | "
                    f"Waiting Time: {total_waiting_time:.2f}s | "
                    f"Vehicles Cleared: {vehicles_cleared} | "
                    f"Vehicles Left: {remaining_vehicles} | "
                    f"Overtime Used: {'Yes' if overtime_used else 'No'} | "
                    f"Cleared All: {'Yes' if cleared_all else 'No'} | "
                    f"Epsilon: {epsilon:.4f} | "
                    f"Route: {os.path.basename(selected_route)}"
                )

                # run validation every N episodes
                if (episode + 1) % self.validation_interval == 0:
                    rl_validation = self.evaluate_controller_on_routes(self.val_routes, controller="rl")
                    validation_wait_history.append((episode + 1, rl_validation["total_waiting_time"]))

                    self.log("---------- VALIDATION CHECK ----------")
                    self.log(f"Episode                      : {episode + 1}")
                    self.log(f"RL Avg Queue                 : {rl_validation['avg_queue']:.4f}")
                    self.log(f"Fixed Avg Queue              : {fixed_baseline['avg_queue']:.4f}")
                    self.log(f"RL Total Waiting            : {rl_validation['total_waiting_time']:.4f}")
                    self.log(f"Fixed Total Waiting         : {fixed_baseline['total_waiting_time']:.4f}")
                    self.log(f"RL Vehicles Cleared         : {rl_validation['vehicles_cleared']:.4f}")
                    self.log(f"Fixed Vehicles Cleared      : {fixed_baseline['vehicles_cleared']:.4f}")
                    self.log(f"RL Vehicles Left            : {rl_validation['remaining_vehicles']:.4f}")
                    self.log(f"Fixed Vehicles Left         : {fixed_baseline['remaining_vehicles']:.4f}")
                    self.log(f"RL Avg Wait/Vehicle         : {rl_validation['avg_wait_per_vehicle']:.4f}")
                    self.log(f"Fixed Avg Wait/Vehicle      : {fixed_baseline['avg_wait_per_vehicle']:.4f}")
                    self.log("--------------------------------------")

                    # save best model by validation waiting time, not by reward
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

    def _get_test_routes(self):
        all_routes = sorted(get_route_files(self.route_dir))
        rng = random.Random(self.hp["random_seed"])
        rng.shuffle(all_routes)

        n_routes = len(all_routes)
        train_end = int(0.70 * n_routes)
        val_end = int(0.85 * n_routes)
        test_routes = all_routes[val_end:]

        if len(test_routes) == 0:
            raise ValueError("Test split is empty. Add more routes or change the split.")

        return test_routes

    def test(self, route_name=None, list_routes=False, model_path=None, test_delay_ms=0):
        """
        Test trained model with GUI.
        """
        self.log("Starting test...")

        test_routes = self._get_test_routes()

        if list_routes:
            self.log("========== AVAILABLE TEST ROUTES ==========")
            for route_file in test_routes:
                self.log(os.path.basename(route_file))
            self.log("===========================================")
            return

        if route_name is not None:
            matching_routes = [
                route_file for route_file in test_routes if os.path.basename(route_file) == route_name
            ]
            if len(matching_routes) == 0:
                available_names = ", ".join(os.path.basename(route_file) for route_file in test_routes[:10])
                raise ValueError(
                    f"Requested test route '{route_name}' is not in the test split. "
                    f"Example available routes: {available_names}"
                )
            selected_route = matching_routes[0]
        else:
            selected_route = random.choice(test_routes)
        self.log(f"Selected test route: {os.path.basename(selected_route)}")

        env = self.create_env(selected_route, use_gui=True)

        observation, info = env.reset()
        state_dim = len(observation)
        action_dim = env.action_space.n

        self.policy_dqn = SumoDQN(state_dim, action_dim, self.fc1_nodes).to(device)
        model_to_load = self._resolve_test_model_file(model_path)
        self.log(f"Loading model file: {model_to_load}")
        self.policy_dqn.load_state_dict(self._load_policy_for_inference(model_to_load))
        self.policy_dqn.eval()
        delay_seconds = max(0.0, float(test_delay_ms) / 1000.0)

        if delay_seconds > 0.0:
            self.log(f"Test pacing enabled: sleeping {delay_seconds:.3f}s after each step")

        done = False
        truncated = False
        total_reward = 0.0
        empty_counter = 0
        overtime_logged = False

        total_queue_normalized_sum = 0.0
        total_queue_raw_sum = 0.0
        total_waiting_time = 0.0
        vehicles_cleared = 0
        step_counter = 0
        prev_total_vehicles = None
        best_remaining_vehicles = float("inf")
        last_progress_sim_step = 0.0

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

            waiting_time = 0.0
            for veh_id in env.sumo.vehicle.getIDList():
                waiting_time += env.sumo.vehicle.getWaitingTime(veh_id)
            total_waiting_time += waiting_time

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
            self.log(f"Executed action         : {action} → {PHASE_MAP.get(action, 'Unknown Phase')}")
            self.log(f"Actual phase after step : {actual_phase_after_step} → {PHASE_MAP.get(actual_phase_after_step, 'Unknown Phase')}")
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

            if delay_seconds > 0.0:
                time.sleep(delay_seconds)

            observation = next_observation

        avg_queue_normalized = total_queue_normalized_sum / max(1, step_counter)
        avg_queue_raw = total_queue_raw_sum / max(1, step_counter)
        avg_wait_per_vehicle = total_waiting_time / max(1, vehicles_cleared)
        remaining_vehicles = self._get_remaining_vehicles(env)
        overtime_used = env.sim_step > self.num_seconds
        cleared_all = remaining_vehicles == 0
        env.close()

        self.log("========== TEST SUMMARY ==========")
        self.log(f"Route file           : {os.path.basename(selected_route)}")
        self.log(f"Total reward         : {total_reward:.8f}")
        self.log(f"Average queue norm   : {avg_queue_normalized:.4f}")
        self.log(f"Average queue raw    : {avg_queue_raw:.4f}")
        self.log(f"Total waiting time   : {total_waiting_time:.4f}")
        self.log(f"Vehicles cleared     : {vehicles_cleared}")
        self.log(f"Vehicles left        : {remaining_vehicles}")
        self.log(f"Overtime used        : {'Yes' if overtime_used else 'No'}")
        self.log(f"Cleared all vehicles : {'Yes' if cleared_all else 'No'}")
        self.log(f"Average wait/vehicle : {avg_wait_per_vehicle:.4f}")
        self.log(f"Simulation steps     : {step_counter}")
        self.log("==================================")


# ============================================================
# BLOCK 9: Script entry point
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SUMO DQN Agent")
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
        default="latest",
        help="Checkpoint path to load for testing, or 'latest' to use the newest saved checkpoint",
    )
    parser.add_argument(
        "--test-delay-ms",
        type=float,
        default=0.0,
        help="Optional delay in milliseconds after each GUI test step",
    )
    args = parser.parse_args()

    hyperparameters = load_hyperparameters(args.hyperparameters)
    agent = Agent(hyperparameters)

    if args.train:
        agent.train(resume_from=args.resume, unseen_only=args.resume_unseen)
    else:
        agent.test(
            route_name=args.route,
            list_routes=args.list_test_routes,
            model_path=args.model,
            test_delay_ms=args.test_delay_ms,
        )
