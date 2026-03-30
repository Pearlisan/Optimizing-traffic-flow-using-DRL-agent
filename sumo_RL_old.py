# ============================================================
# BLOCK 1: Import libraries
# ============================================================
import argparse
import os
import random
from datetime import datetime

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
    with open(yaml_file, "r") as file:
        configs = yaml.safe_load(file)

    if config_name not in configs:
        raise ValueError(f"Config '{config_name}' not found in {yaml_file}")

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
# BLOCK 7: Paper-style reward function for SUMO-RL
# ============================================================
def paper_reward(ts):
    """
    Reward inspired by the paper:

        reward = passing_vehicles / stopped_vehicles

    Practical implementation details:
    - passing_vehicles is approximated as the number of NEW vehicle IDs
      observed on outgoing lanes since the last control step.
    - stopped_vehicles is the current number of queued/stopped vehicles.
    - denominator is clamped with max(1, stopped_vehicles) to avoid
      division by zero explosions.
    """

    # Collect all vehicle IDs currently present on outgoing lanes
    current_outgoing_ids = set()
    for lane in ts.out_lanes:
        lane_vehicle_ids = ts.sumo.lane.getLastStepVehicleIDs(lane)
        current_outgoing_ids.update(lane_vehicle_ids)

    # Initialize memory on first call
    if not hasattr(ts, "_prev_outgoing_ids"):
        ts._prev_outgoing_ids = set()

    # Vehicles newly appearing on outgoing lanes during this interval
    new_outgoing_ids = current_outgoing_ids - ts._prev_outgoing_ids
    passed_vehicles = len(new_outgoing_ids)

    # Current stopped / queued vehicles
    stopped_vehicles = ts.get_total_queued()

    # Paper-style reward with safe denominator
    reward = passed_vehicles / max(1, stopped_vehicles)

    # Update memory for next interval
    ts._prev_outgoing_ids = current_outgoing_ids

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
        self.net_file = self.hp["net_file"]
        self.route_dir = self.hp["route_dir"]

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

        # Early-stop hyperparameters
        self.early_stop_patience = self.hp.get("early_stop_patience", 3)

        # Runtime paths
        self.runs_dir = "runs"
        os.makedirs(self.runs_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.timestamp = timestamp

        self.log_file = os.path.join(self.runs_dir, f"sumo1_{timestamp}.log")
        self.graph_file = os.path.join(self.runs_dir, f"sumo1_{timestamp}.png")
        self.model_file = os.path.join(self.runs_dir, "sumo1.pt")

        # Loss function
        self.loss_fn = nn.MSELoss()

    def log(self, message):
        """
        Print to console and save to log file.
        """
        print(message)
        with open(self.log_file, "a", encoding="utf-8") as file:
            file.write(message + "\n")

    def create_env(self, route_file, use_gui=False):
        """
        Create SUMO environment using the paper-style reward.
        """
        env = SumoEnvironment(
            net_file=self.net_file,
            route_file=route_file,
            use_gui=use_gui,
            num_seconds=self.num_seconds,
            delta_time=self.delta_time,
            yellow_time=self.yellow_time,
            min_green=self.min_green,
            single_agent=True,
            reward_fn=paper_reward,
            sumo_warnings=self.sumo_warnings,
            additional_sumo_cmd=self.additional_sumo_cmd if use_gui else ""
        )
        return env

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

    def save_graph(self, rewards_per_episode, epsilon_history):
        """
        Save reward curve plot and epsilon decay plot on same figure.
        """
        fig, ax1 = plt.subplots(figsize=(9, 5))

        rewards_array = np.array(rewards_per_episode)
        moving_avg = np.zeros(len(rewards_array))

        for i in range(len(rewards_array)):
            moving_avg[i] = np.mean(rewards_array[max(0, i - 9):i + 1])

        ax1.plot(rewards_array, label="Episode Reward")
        ax1.plot(moving_avg, label="10-Episode Moving Average")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Reward")
        ax1.set_title("SUMO DQN Training Rewards + Epsilon Decay")
        ax1.grid(True)

        ax2 = ax1.twinx()
        ax2.plot(epsilon_history, linestyle="--", label="Epsilon")
        ax2.set_ylabel("Epsilon")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

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

    def train(self):
        """
        Train DQN agent.
        """
        self.log("Starting training...")

        # get all route files and create a reproducible split
        all_routes = sorted(get_route_files(self.route_dir))
        random.Random(self.hp["random_seed"]).shuffle(all_routes)

        train_routes = all_routes[:600]
        test_routes = all_routes[600:1200]

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
        best_reward = -float("inf")
        rewards_per_episode = []
        epsilon_history = []

        episode = 0

        # keep training until num_episodes is reached
        while episode < self.num_episodes:

            # shuffle once per cycle, then iterate through all routes exactly once
            random.shuffle(train_routes)

            for selected_route in train_routes:
                if episode >= self.num_episodes:
                    break

                env = self.create_env(selected_route, use_gui=False)

                observation, info = env.reset()
                state = torch.tensor(observation, dtype=torch.float32, device=device)

                done = False
                truncated = False
                episode_reward = 0.0
                empty_counter = 0

                # metrics tracking
                total_queue_sum = 0.0
                total_waiting_time = 0.0
                vehicles_cleared = 0
                step_counter = 0
                prev_total_vehicles = None

                while not (done or truncated):
                    action = self.select_action(
                        state_tensor=state,
                        observation=observation,
                        env=env,
                        epsilon=epsilon,
                        is_training=True
                    )

                    next_observation, reward, done, truncated, info = env.step(action)
                    next_state = torch.tensor(next_observation, dtype=torch.float32, device=device)

                    # queue metric
                    current_queue = sum(next_observation[13:21])
                    total_queue_sum += current_queue

                    # waiting time metric
                    waiting_time = 0.0
                    for veh_id in env.sumo.vehicle.getIDList():
                        waiting_time += env.sumo.vehicle.getWaitingTime(veh_id)
                    total_waiting_time += waiting_time

                    # vehicles cleared metric
                    current_total_vehicles = env.sumo.simulation.getMinExpectedNumber()
                    if prev_total_vehicles is not None:
                        cleared = max(0, prev_total_vehicles - current_total_vehicles)
                        vehicles_cleared += cleared
                    prev_total_vehicles = current_total_vehicles

                    step_counter += 1

                    if self._traffic_is_finished(env):
                        empty_counter += 1
                    else:
                        empty_counter = 0

                    manual_terminal = empty_counter >= self.early_stop_patience

                    self.memory.append(
                        (state, action, next_state, reward, done or truncated or manual_terminal)
                    )

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

                env.close()

                avg_queue = total_queue_sum / max(1, step_counter)
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)

                rewards_per_episode.append(episode_reward)
                epsilon_history.append(epsilon)

                self.log(
                    f"Episode {episode + 1:03d} | "
                    f"Reward: {episode_reward:.2f} | "
                    f"Avg Queue: {avg_queue:.2f} | "
                    f"Waiting Time: {total_waiting_time:.2f}s | "
                    f"Vehicles Cleared: {vehicles_cleared} | "
                    f"Epsilon: {epsilon:.4f} | "
                    f"Route: {os.path.basename(selected_route)}"
                )

                if episode_reward > best_reward:
                    best_reward = episode_reward
                    torch.save(self.policy_dqn.state_dict(), self.model_file)
                    self.log(f"New best model saved. Best reward = {best_reward:.8f}")

                if (episode + 1) % 10 == 0:
                    self.save_graph(rewards_per_episode, epsilon_history)

                episode += 1

        self.save_graph(rewards_per_episode, epsilon_history)
        self.log("Training finished.")

    def test(self):
        """
        Test trained model with GUI.
        """
        self.log("Starting test...")

        # get all the route files
        all_routes = sorted(get_route_files(self.route_dir))
        random.Random(self.hp["random_seed"]).shuffle(all_routes)

        # split into training and testing
        train_routes = all_routes[:600]
        test_routes = all_routes[600:1200]

        # select a random file from the test routes
        selected_route = random.choice(test_routes)
        self.log(f"Selected test route: {os.path.basename(selected_route)}")

        env = self.create_env(selected_route, use_gui=True)

        observation, info = env.reset()
        state_dim = len(observation)
        action_dim = env.action_space.n

        self.policy_dqn = SumoDQN(state_dim, action_dim, self.fc1_nodes).to(device)
        self.policy_dqn.load_state_dict(torch.load(self.model_file, map_location=device))
        self.policy_dqn.eval()

        done = False
        truncated = False
        total_reward = 0.0
        empty_counter = 0

        # benchmark metrics
        total_queue_sum = 0.0
        total_waiting_time = 0.0
        vehicles_cleared = 0
        step_counter = 0
        prev_total_vehicles = None

        while not (done or truncated):
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

            # queue metric
            current_queue = sum(next_observation[13:21])
            total_queue_sum += current_queue

            # waiting time metric
            waiting_time = 0.0
            for veh_id in env.sumo.vehicle.getIDList():
                waiting_time += env.sumo.vehicle.getWaitingTime(veh_id)
            total_waiting_time += waiting_time

            # vehicles cleared metric
            current_total_vehicles = env.sumo.simulation.getMinExpectedNumber()
            if prev_total_vehicles is not None:
                vehicles_cleared += max(0, prev_total_vehicles - current_total_vehicles)
            prev_total_vehicles = current_total_vehicles

            step_counter += 1
            total_reward += reward

            self.log(f"\n[Step {env.sim_step}]")
            self.log(f"Executed action         : {action} → {PHASE_MAP.get(action, 'Unknown Phase')}")
            self.log(f"Actual phase after step : {actual_phase_after_step} → {PHASE_MAP.get(actual_phase_after_step, 'Unknown Phase')}")
            self.log(f"Phase one-hot           : {next_observation[:4]}")
            self.log(f"Min green               : {next_observation[4]}")
            self.log(f"Queues                  : {next_observation[13:21]}")
            self.log(f"Reward                  : {reward:.8f}")
            self.log(f"Info                    : {info}")

            if self._traffic_is_finished(env):
                empty_counter += 1
            else:
                empty_counter = 0

            if empty_counter >= self.early_stop_patience:
                self.log(f"Test early-stopped at sim step {env.sim_step} because traffic cleared.")
                break

            observation = next_observation

        env.close()

        avg_queue = total_queue_sum / max(1, step_counter)
        avg_wait_per_vehicle = total_waiting_time / max(1, vehicles_cleared)

        self.log("========== TEST SUMMARY ==========")
        self.log(f"Route file           : {os.path.basename(selected_route)}")
        self.log(f"Total reward         : {total_reward:.8f}")
        self.log(f"Average queue        : {avg_queue:.4f}")
        self.log(f"Total waiting time   : {total_waiting_time:.4f}")
        self.log(f"Vehicles cleared     : {vehicles_cleared}")
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
    args = parser.parse_args()

    hyperparameters = load_hyperparameters(args.hyperparameters)
    agent = Agent(hyperparameters)

    if args.train:
        agent.train()
    else:
        agent.test()