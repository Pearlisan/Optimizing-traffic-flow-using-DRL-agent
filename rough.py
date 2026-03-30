# ============================================================
# BLOCK 1: Import libraries
# ============================================================
import gymnasium as gym
import sumo_rl
from sumo_rl import SumoEnvironment
import os
import random
from datetime import datetime

# ============================================================
# BLOCK 1.5: Logger setup (prints + saves to file)
# ============================================================
# LOG_FILE = "fixed_time_model.log"
LOG_FILE = f"fixed_time_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

def log(message):
    print(message)  
    # Add encoding="utf-8" here
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(str(message) + "\n")
# ============================================================
# BLOCK 2: Set working directory and file paths
# ============================================================
os.chdir(r"C:\Users\Pearlisan\Documents\Sumo\test\crossnetwork_py")

NET_FILE = r"dataset\net\net_single2.net.xml"
ROUTE_DIR = r"dataset\rou\net_single2\c1"


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
# BLOCK 4: Collect all route files in the selected folder
# ============================================================
route_files = [
    os.path.join(ROUTE_DIR, f)
    for f in os.listdir(ROUTE_DIR)
    if f.endswith(".rou.xml")
]


# ============================================================
# BLOCK 5: Helper function to decode current phase from observation
# ============================================================
def get_current_phase_from_obs(observation):
    """
    Extract the current green phase index from the one-hot encoded
    phase vector at the start of the observation.
    """
    phase_one_hot = observation[:4]
    return int(phase_one_hot.argmax())


# ============================================================
# BLOCK 6: Run one SUMO episode
# ============================================================
# ============================================================
# BLOCK 6: Run one SUMO episode
# ============================================================
# ============================================================
# BLOCK 6: Run one SUMO episode
# ============================================================

# ============================================================
# BLOCK 6: Run one SUMO episode
# ============================================================
def run_one_episode(route_file):
    log(f"\nUsing route file: {route_file}")

    env = SumoEnvironment(
        net_file=NET_FILE,
        route_file=route_file,
        use_gui=True,
        num_seconds=3600,
        delta_time=5,
        min_green=20,
        single_agent=True,
        additional_sumo_cmd="--delay 1"
    )

    observation, info = env.reset()

    log("\nInitial State:")
    log(observation)
    log(f"State length: {len(observation)}")
    log(f"Initial phase one-hot: {observation[:4]}")
    log(f"Initial min_green: {observation[4]}")

    # ============================================================
    # BLOCK 6.1: Get SUMO / TraCI handles
    # ============================================================
    # sumo-rl keeps the live traci connection here
    traci_conn = env.sumo

    # Get the single traffic signal id
    ts_id = list(env.traffic_signals.keys())[0]

    # Controlled incoming lanes for this signal
    incoming_lanes = env.traffic_signals[ts_id].lanes

    done = False
    truncated = False

    # ============================================================
    # BLOCK 6.2: Initialize true summary trackers
    # ============================================================
    total_reward = 0.0
    queue_sum = 0.0
    step_count = 0
    total_waiting_time = 0.0
    vehicles_cleared = 0

    while not (done or truncated):
        # Read current phase and min_green BEFORE taking action
        current_phase = get_current_phase_from_obs(observation)
        min_green_flag = int(observation[4])

        # Agent proposes a random action
        proposed_action = env.action_space.sample()

        # If minimum green time has NOT passed, force the action
        # to stay on the current phase
        if min_green_flag == 0:
            action = current_phase
        else:
            action = proposed_action

        # Step environment
        next_observation, reward_from_env, done, truncated, info = env.step(action)

        # ============================================================
        # BLOCK 6.3: Compute TRUE metrics from SUMO, not observation
        # ============================================================
        # True queue = number of halting vehicles on controlled incoming lanes
        step_queue = 0
        lane_vehicle_ids = []

        for lane_id in incoming_lanes:
            step_queue += traci_conn.lane.getLastStepHaltingNumber(lane_id)
            lane_vehicle_ids.extend(traci_conn.lane.getLastStepVehicleIDs(lane_id))

        # Remove duplicates in case a vehicle appears more than once
        lane_vehicle_ids = list(set(lane_vehicle_ids))

        # True waiting = sum of accumulated waiting times of vehicles currently on incoming lanes
        step_waiting_time = 0.0
        for veh_id in lane_vehicle_ids:
            step_waiting_time += traci_conn.vehicle.getAccumulatedWaitingTime(veh_id)

        # Vehicles cleared during this step
        step_arrived = traci_conn.simulation.getArrivedNumber()

        # Reward based on TRUE queue count
        # This makes the fixed-controller summary comparable and meaningful
        reward = -float(step_queue)

        total_reward += reward
        queue_sum += step_queue
        total_waiting_time += step_waiting_time
        vehicles_cleared += step_arrived
        step_count += 1

        # Decode the ACTUAL phase after stepping
        actual_phase_after_step = get_current_phase_from_obs(next_observation)

        log(f"\n[Step {env.sim_step}]")
        log(f"Current phase before step : {current_phase} → {PHASE_MAP.get(current_phase, 'Unknown Phase')}")
        log(f"Proposed action           : {proposed_action} → {PHASE_MAP.get(proposed_action, 'Unknown Phase')}")
        log(f"Executed action           : {action} → {PHASE_MAP.get(action, 'Unknown Phase')}")
        log(f"Actual phase after step   : {actual_phase_after_step} → {PHASE_MAP.get(actual_phase_after_step, 'Unknown Phase')}")
        log(f"Phase one-hot             : {next_observation[:4]}")
        log(f"Min green                 : {next_observation[4]}")
        log(f"Densities                 : {next_observation[5:13]}")
        log(f"Queues                    : {next_observation[13:21]}")
        log(f"Reward from env           : {reward_from_env}")
        log(f"True queue (halting vehs) : {step_queue}")
        log(f"Step waiting time         : {step_waiting_time:.4f}")
        log(f"Vehicles arrived this step: {step_arrived}")
        log(f"Summary reward used       : {reward}")

        observation = next_observation

    # ============================================================
    # BLOCK 6.4: Final summary
    # ============================================================
    average_queue = queue_sum / step_count if step_count > 0 else 0.0
    average_wait_per_vehicle = (
        total_waiting_time / vehicles_cleared if vehicles_cleared > 0 else 0.0
    )

    log("\n========== TEST SUMMARY ==========")
    log(f"Route file           : {os.path.basename(route_file)}")
    log(f"Total reward         : {total_reward:.8f}")
    log(f"Average queue        : {average_queue:.4f}")
    log(f"Total waiting time   : {total_waiting_time:.4f}")
    log(f"Vehicles cleared     : {vehicles_cleared}")
    log(f"Average wait/vehicle : {average_wait_per_vehicle:.4f}")
    log(f"Simulation steps     : {step_count}")
    log("==================================")

    env.close()


# ============================================================
# BLOCK 7: Main function
# ============================================================
def main():
    # Also add encoding="utf-8" here when clearing the file
    open(LOG_FILE, "w", encoding="utf-8").close()
    
    selected_route = os.path.join(ROUTE_DIR, "_106_18_17_104.rou.xml")
    run_one_episode(selected_route)

# ============================================================
# BLOCK 8: Script entry point
# ============================================================
if __name__ == "__main__":
    main()