# This block imports TraCI and starts your SUMO simulation.
import traci

SUMO_BINARY = "sumo-gui"
# SUMO_CFG = "crossnetwork.sumocfg"
SUMO_CFG = r"C:\Users\Pearlisan\Documents\Sumo\test\crossnetwork.sumocfg"
TLS_ID = "0"

sumo_cmd = [SUMO_BINARY, "-c", SUMO_CFG]
traci.start(sumo_cmd)

# Lane groups
LANE_GROUPS = {
    "W_left":    ["1i_1"],
    "W_through": ["1i_0"],

    "E_left":    ["2i_1"],
    "E_through": ["2i_0"],

    "S_left":    ["3i_2"],
    "S_through": ["3i_1"],

    "N_left":    ["4i_2"],
    "N_through": ["4i_1"],
}

# Queue length helper
def get_lane_group_queue(lane_ids):
    total_queue = 0
    for lane_id in lane_ids:
        total_queue += traci.lane.getLastStepHaltingNumber(lane_id)
    return total_queue

# 8 queue values
def get_queue_state():
    return (
        get_lane_group_queue(LANE_GROUPS["W_left"]),
        get_lane_group_queue(LANE_GROUPS["W_through"]),

        get_lane_group_queue(LANE_GROUPS["E_left"]),
        get_lane_group_queue(LANE_GROUPS["E_through"]),

        get_lane_group_queue(LANE_GROUPS["S_left"]),
        get_lane_group_queue(LANE_GROUPS["S_through"]),

        get_lane_group_queue(LANE_GROUPS["N_left"]),
        get_lane_group_queue(LANE_GROUPS["N_through"]),
    )

# Convert SUMO phase (0–7) → RL phase (0–3)
def get_rl_phase():
    phase_index = traci.trafficlight.getPhase(TLS_ID)
    rl_phase = phase_index // 2
    return rl_phase

# Simulation loop
step = 0

while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()

    queues = get_queue_state()
    phase = get_rl_phase()

    print(f"Step {step}: queues={queues}, phase={phase}")

    step += 1

traci.close()