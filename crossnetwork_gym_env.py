# This block imports the libraries used by the Gym environment.
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import traci


# This block defines a SUMO traffic-signal Gym environment.
class SumoTrafficEnv(gym.Env):
    def __init__(self, sumo_cfg="crossnetwork_py/crossnetwork.sumocfg", use_gui=True):
        super().__init__()

        # This block stores SUMO settings.
        self.sumo_cfg = sumo_cfg
        self.sumo_binary = "sumo-gui" if use_gui else "sumo"
        self.tls_id = "0"

        # This block defines the 4 logical green phases.
        # 0 = NS through, 1 = EW through, 2 = NS left, 3 = EW left
        self.action_space = spaces.Discrete(4)

        # This block defines timing settings.
        self.control_steps = 10          # decision interval
        self.yellow_steps = 5            # yellow duration before switching
        self.min_green_steps = 10        # do not switch before this
        self.max_green_steps = 60        # must switch after this

        # This block defines scaling factors for normalized features.
        self.queue_norm_cap = 50.0
        self.phase_demand_norm_cap = 100.0
        self.phase_timer_norm_cap = float(self.max_green_steps)
        self.starvation_norm_cap = 120.0

        # This block defines reward weights.
        self.starvation_penalty_weight = 15.0

        # This block defines the lane groups used to measure queues.
        self.lane_groups = {
            "W_left":    ["1i_1"],
            "W_through": ["1i_0"],
            "E_left":    ["2i_1"],
            "E_through": ["2i_0"],
            "S_left":    ["3i_2"],
            "S_through": ["3i_1"],
            "N_left":    ["4i_2"],
            "N_through": ["4i_1"],
        }

        # This block maps RL phases to SUMO green phases.
        self.green_phases = {
            0: 0,   # NS through green
            1: 2,   # EW through green
            2: 4,   # NS left green
            3: 6,   # EW left green
        }

        # This block maps RL phases to SUMO yellow phases.
        self.yellow_phases = {
            0: 1,
            1: 3,
            2: 5,
            3: 7,
        }

        # This block defines the observation:
        # 8 normalized queues
        # 4 current-phase one-hot values
        # 4 normalized phase-service-demand values
        # 1 normalized time in current phase
        # 4 normalized starvation timers
        # total = 21 features
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(21,),
            dtype=np.float32
        )

        # This block initializes episode counters.
        self.max_steps = 1000
        self.step_count = 0
        self.current_rl_phase = 0
        self.time_in_current_phase = 0
        self.phase_starvation_timers = [0, 0, 0, 0]

    # This block starts SUMO.
    def _start_sumo(self):
        sumo_cmd = [
            self.sumo_binary,
            "-c", self.sumo_cfg,
            "--start",
            "--delay", "100"
        ]
        traci.start(sumo_cmd)

    # This block returns the halted queue count for a lane group.
    def _get_lane_group_queue(self, lane_ids):
        total_queue = 0
        for lane_id in lane_ids:
            total_queue += traci.lane.getLastStepHaltingNumber(lane_id)
        return total_queue

    # This block returns the 8 raw queue values in fixed order.
    def _get_queue_state(self):
        return [
            self._get_lane_group_queue(self.lane_groups["W_left"]),
            self._get_lane_group_queue(self.lane_groups["W_through"]),
            self._get_lane_group_queue(self.lane_groups["E_left"]),
            self._get_lane_group_queue(self.lane_groups["E_through"]),
            self._get_lane_group_queue(self.lane_groups["S_left"]),
            self._get_lane_group_queue(self.lane_groups["S_through"]),
            self._get_lane_group_queue(self.lane_groups["N_left"]),
            self._get_lane_group_queue(self.lane_groups["N_through"]),
        ]

    # This block converts the current SUMO phase into one of the 4 logical phases.
    def _get_rl_phase(self):
        sumo_phase = traci.trafficlight.getPhase(self.tls_id)
        return sumo_phase // 2

    # This block clips and normalizes any value to [0,1].
    def _normalize(self, value, cap):
        return min(float(value) / float(cap), 1.0)

    # This block computes service demand for each logical phase.
    def _get_phase_service_demand(self, queues):
        W_left, W_through, E_left, E_through, S_left, S_through, N_left, N_through = queues

        return [
            S_through + N_through,  # phase 0 serves NS through
            W_through + E_through,  # phase 1 serves EW through
            S_left + N_left,        # phase 2 serves NS left
            W_left + E_left,        # phase 3 serves EW left
        ]

    # This block builds the full normalized state vector.
    def _get_state(self):
        queues = self._get_queue_state()
        phase_demands = self._get_phase_service_demand(queues)

        normalized_queues = [self._normalize(q, self.queue_norm_cap) for q in queues]

        current_phase_vector = [0.0, 0.0, 0.0, 0.0]
        current_phase_vector[self.current_rl_phase] = 1.0

        normalized_phase_demands = [
            self._normalize(d, self.phase_demand_norm_cap) for d in phase_demands
        ]

        normalized_phase_timer = [
            self._normalize(self.time_in_current_phase, self.phase_timer_norm_cap)
        ]

        normalized_starvation = [
            self._normalize(t, self.starvation_norm_cap) for t in self.phase_starvation_timers
        ]

        state = np.array(
            normalized_queues
            + current_phase_vector
            + normalized_phase_demands
            + normalized_phase_timer
            + normalized_starvation,
            dtype=np.float32
        )
        return state

    # This block runs SUMO forward by n simulation steps.
    def _run_steps(self, n):
        for _ in range(n):
            traci.simulationStep()

    # This block updates starvation timers after a control decision.
    def _update_starvation_timers(self, served_phase):
        for phase in range(4):
            if phase == served_phase:
                self.phase_starvation_timers[phase] = 0
            else:
                self.phase_starvation_timers[phase] += self.control_steps

    # This block selects a fallback phase when max green is reached.
    def _choose_forced_phase(self, requested_phase):
        if requested_phase != self.current_rl_phase:
            return requested_phase

        queues = self._get_queue_state()
        phase_demands = self._get_phase_service_demand(queues)

        candidate_phases = [p for p in range(4) if p != self.current_rl_phase]

        # This score favors large demand and phases starved for long.
        scores = {}
        for p in candidate_phases:
            demand_score = phase_demands[p]
            starvation_score = 0.5 * self.phase_starvation_timers[p]
            scores[p] = demand_score + starvation_score

        best_phase = max(scores, key=scores.get)
        return best_phase

    # This block handles phase holding, yellow switching, and green timing.
    def _apply_action_with_timing(self, requested_phase):
        requested_phase = int(requested_phase)

        # This block enforces minimum green time.
        if self.time_in_current_phase < self.min_green_steps:
            chosen_phase = self.current_rl_phase

            green_index = self.green_phases[chosen_phase]
            traci.trafficlight.setPhase(self.tls_id, green_index)
            self._run_steps(self.control_steps)

            self.time_in_current_phase += self.control_steps
            self._update_starvation_timers(chosen_phase)
            return chosen_phase, "held_by_min_green"

        # This block enforces maximum green time.
        if self.time_in_current_phase >= self.max_green_steps:
            chosen_phase = self._choose_forced_phase(requested_phase)
        else:
            chosen_phase = requested_phase

        # This block keeps the same phase if no switch is needed.
        if chosen_phase == self.current_rl_phase:
            green_index = self.green_phases[chosen_phase]
            traci.trafficlight.setPhase(self.tls_id, green_index)
            self._run_steps(self.control_steps)

            self.time_in_current_phase += self.control_steps
            self._update_starvation_timers(chosen_phase)
            return chosen_phase, "kept_same_phase"

        # This block inserts yellow before switching to the new green.
        current_yellow_index = self.yellow_phases[self.current_rl_phase]
        traci.trafficlight.setPhase(self.tls_id, current_yellow_index)
        self._run_steps(self.yellow_steps)

        new_green_index = self.green_phases[chosen_phase]
        traci.trafficlight.setPhase(self.tls_id, new_green_index)
        self._run_steps(self.control_steps)

        self.current_rl_phase = chosen_phase
        self.time_in_current_phase = self.control_steps
        self._update_starvation_timers(chosen_phase)

        return chosen_phase, "switched_phase"

    # This block computes the reward with a starvation penalty.
    def _get_reward(self):
        queues = self._get_queue_state()
        total_queue = float(sum(queues))

        phase_demands = self._get_phase_service_demand(queues)

        starvation_penalty = 0.0
        for phase in range(4):
            demand_norm = self._normalize(phase_demands[phase], self.phase_demand_norm_cap)
            starve_norm = self._normalize(self.phase_starvation_timers[phase], self.starvation_norm_cap)
            starvation_penalty += demand_norm * starve_norm

        reward = -total_queue - self.starvation_penalty_weight * starvation_penalty

        reward_details = {
            "total_queue": total_queue,
            "starvation_penalty": starvation_penalty,
            "final_reward": reward,
        }
        return reward, reward_details

    # This block resets the SUMO simulation and environment state.
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        try:
            traci.close()
        except:
            pass

        self._start_sumo()

        self.step_count = 0
        self.current_rl_phase = self._get_rl_phase()
        self.time_in_current_phase = 0
        self.phase_starvation_timers = [0, 0, 0, 0]

        state = self._get_state()
        info = {
            "rl_phase": self.current_rl_phase,
            "sumo_phase": traci.trafficlight.getPhase(self.tls_id),
            "time_in_current_phase": self.time_in_current_phase,
            "phase_starvation_timers": self.phase_starvation_timers.copy(),
        }
        return state, info

    # This block executes one environment step.
    def step(self, action):
        chosen_phase, phase_decision_note = self._apply_action_with_timing(action)

        self.step_count += 1

        state = self._get_state()
        reward, reward_details = self._get_reward()

        queues = self._get_queue_state()
        phase_demands = self._get_phase_service_demand(queues)

        terminated = traci.simulation.getMinExpectedNumber() <= 0
        truncated = self.step_count >= self.max_steps

        info = {
            "requested_action": int(action),
            "chosen_phase": chosen_phase,
            "decision_note": phase_decision_note,
            "sumo_phase": traci.trafficlight.getPhase(self.tls_id),
            "rl_phase": self.current_rl_phase,
            "raw_queues": queues,
            "phase_service_demand": phase_demands,
            "time_in_current_phase": self.time_in_current_phase,
            "phase_starvation_timers": self.phase_starvation_timers.copy(),
            "reward_total_queue": reward_details["total_queue"],
            "reward_starvation_penalty": reward_details["starvation_penalty"],
            "reward_final": reward_details["final_reward"],
        }

        return state, reward, terminated, truncated, info

    # This block closes the TraCI connection.
    def close(self):
        try:
            traci.close()
        except:
            pass


# This block tests the Gym environment without any RL model.
env = SumoTrafficEnv(
    sumo_cfg="crossnetwork_py/crossnetwork.sumocfg",
    use_gui=True
)

state, info = env.reset()

print("\n" + "=" * 80)
print("INITIAL ENVIRONMENT STATE")
print("=" * 80)
print("State vector shape:", state.shape)
print("State vector:", state)
print("Initial RL phase:", info["rl_phase"])
print("Initial SUMO phase index:", info["sumo_phase"])
print("Time in current phase:", info["time_in_current_phase"])
print("Starvation timers:", info["phase_starvation_timers"])
print("=" * 80)

for step in range(100):
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(action)

    print(f"\nSTEP {step}")
    print("-" * 80)
    print(f"Requested action           : {info['requested_action']}")
    print(f"Chosen phase               : {info['chosen_phase']}")
    print(f"Decision note              : {info['decision_note']}")
    print(f"SUMO phase index           : {info['sumo_phase']}")
    print(f"Current RL phase           : {info['rl_phase']}")
    print(f"Time in current phase      : {info['time_in_current_phase']}")
    print(f"Raw queues                 : {info['raw_queues']}")
    print(f"Phase service demand       : {info['phase_service_demand']}")
    print(f"Phase starvation timers    : {info['phase_starvation_timers']}")
    print(f"Reward from total queue    : {-info['reward_total_queue']}")
    print(f"Starvation penalty term    : {-env.starvation_penalty_weight * info['reward_starvation_penalty']:.4f}")
    print(f"Final reward               : {reward:.4f}")
    print(f"Next state shape           : {next_state.shape}")
    print(f"Next state vector          : {next_state}")
    print("-" * 80)

    if terminated:
        print("\nSimulation ended because no more vehicles are expected.")
        break

    if truncated:
        print("\nEpisode stopped because max_steps was reached.")
        break

env.close()