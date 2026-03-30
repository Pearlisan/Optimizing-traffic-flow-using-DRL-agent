# ============================================================
# BLOCK 1: Import libraries
# ============================================================
import os
import csv
import traci
from datetime import datetime
import matplotlib.pyplot as plt
from sumolib import checkBinary


# ============================================================
# BLOCK 2: Set SUMO file paths
# ============================================================
NET_FILE = r"crossnetwork_py\dataset\net\net_single2.net.xml"
ROUTE_FILE = r"crossnetwork_py\dataset\rou\net_single2\c2\_39_68_127_26.rou.xml"

# Change this to your traffic light id if needed
TLS_ID = "t"
GUI_DELAY_MS = 100
GUI_SCHEMA = "real world"


# ============================================================
# BLOCK 3: Create log file
# ============================================================
runs_dir = "runs"
os.makedirs(runs_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(runs_dir, f"fixed_test_{timestamp}.log")


# ============================================================
# BLOCK 4: Logging helper
# ============================================================
def log(message):
    """
    Print message to terminal and append it to a log file.
    """
    print(message)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def get_directional_queue_raw(controlled_lanes):
    ns_queue_raw = 0.0
    we_queue_raw = 0.0
    for lane_id in controlled_lanes:
        lane_queue = float(traci.lane.getLastStepHaltingNumber(lane_id))
        if lane_id.startswith(("n_", "s_")):
            ns_queue_raw += lane_queue
        elif lane_id.startswith(("e_", "w_")):
            we_queue_raw += lane_queue
    return ns_queue_raw, we_queue_raw


def write_directional_queue_outputs(csv_path, png_path, queue_rows):
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["step_index", "sim_step", "ns_queue_raw", "we_queue_raw"])
        writer.writerows(queue_rows)

    ns_values = [row[2] for row in queue_rows]
    we_values = [row[3] for row in queue_rows]

    plt.figure(figsize=(9, 5))
    bins = range(0, int(max(ns_values + we_values + [1])) + 2)
    plt.hist(ns_values, bins=bins, density=True, alpha=0.35, color="red", label=f"NS, mean = {sum(ns_values) / max(1, len(ns_values)):.2f}")
    plt.hist(we_values, bins=bins, density=True, alpha=0.35, color="royalblue", label=f"WE, mean = {sum(we_values) / max(1, len(we_values)):.2f}")
    plt.xlabel("Queue length bins")
    plt.ylabel("Normalized Frequency")
    plt.title("Histogram and distribution of queue lengths for fixed-time controller")
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()


# ============================================================
# BLOCK 5: Build SUMO command
# ============================================================
sumo_binary = checkBinary("sumo-gui")

sumo_cmd = [
    sumo_binary,
    "-n", NET_FILE,
    "-r", ROUTE_FILE,
    "--start",
    "--delay", str(GUI_DELAY_MS),
    "--quit-on-end",
    "--duration-log.disable", "true",
    "--no-step-log", "true",
]


# ============================================================
# BLOCK 6: Run simulation and compute summary metrics
# ============================================================
def run_fixed_simulation():
    """
    Run fixed-time SUMO simulation, log live step metrics,
    and stop as soon as all traffic has truly finished.
    """
    traci.start(sumo_cmd)

    # Use a human-friendly vehicle rendering mode after the GUI starts.
    traci.gui.setSchema("View #0", GUI_SCHEMA)

    step_count = 0
    total_reward = 0.0
    queue_sum = 0.0
    total_waiting_time = 0.0
    total_co2_emission = 0.0
    vehicles_cleared = 0
    overtime_logged = False
    directional_queue_rows = []

    # Track whether traffic has started, so we do not stop too early
    has_seen_vehicle = False
    prev_total_vehicles = None

    # Use all lanes controlled by the traffic light
    controlled_lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(TLS_ID)))

    log("Starting fixed-time simulation...")
    log(f"Network file : {NET_FILE}")
    log(f"Route file   : {ROUTE_FILE}")
    log(f"Traffic light: {TLS_ID}")
    log(f"GUI delay    : {GUI_DELAY_MS} ms")
    log(f"GUI schema   : {GUI_SCHEMA}")
    log("")

    while True:
        traci.simulationStep()
        step_count += 1

        # Current simulation-wide counts
        active_vehicles = traci.vehicle.getIDCount()
        remaining_vehicles = traci.simulation.getMinExpectedNumber()

        if active_vehicles > 0:
            has_seen_vehicle = True

        # ------------------------------------------------------------
        # BLOCK 6.1: True queue from halting vehicles on controlled lanes
        # ------------------------------------------------------------
        step_queue = 0
        lane_vehicle_ids = []

        for lane_id in controlled_lanes:
            step_queue += traci.lane.getLastStepHaltingNumber(lane_id)
            lane_vehicle_ids.extend(traci.lane.getLastStepVehicleIDs(lane_id))

        lane_vehicle_ids = list(set(lane_vehicle_ids))
        ns_queue_raw, we_queue_raw = get_directional_queue_raw(controlled_lanes)
        directional_queue_rows.append((step_count, step_count, ns_queue_raw, we_queue_raw))

        # ------------------------------------------------------------
        # BLOCK 6.2: Waiting time from all vehicles currently in the network
        # ------------------------------------------------------------
        step_waiting_time = 0.0
        for veh_id in traci.vehicle.getIDList():
            step_waiting_time += traci.vehicle.getWaitingTime(veh_id)

        step_co2_emission = 0.0
        for veh_id in traci.vehicle.getIDList():
            step_co2_emission += traci.vehicle.getCO2Emission(veh_id)

        # ------------------------------------------------------------
        # BLOCK 6.3: Vehicles cleared this step
        # ------------------------------------------------------------
        step_arrived = 0
        if prev_total_vehicles is not None:
            step_arrived = max(0, prev_total_vehicles - remaining_vehicles)
            vehicles_cleared += step_arrived
        prev_total_vehicles = remaining_vehicles

        # ------------------------------------------------------------
        # BLOCK 6.4: Reward based on queue
        # ------------------------------------------------------------
        reward = -float(step_queue)

        total_reward += reward
        queue_sum += step_queue
        total_waiting_time += step_waiting_time
        total_co2_emission += step_co2_emission

        # ------------------------------------------------------------
        # BLOCK 6.5: Live logging
        # ------------------------------------------------------------
        if step_count >= 1200 and remaining_vehicles > 0 and not overtime_logged:
            overtime_logged = True
            log(
                f"Fixed test exceeded nominal duration 1200s and is continuing until "
                f"clearance. Vehicles left: {remaining_vehicles}"
            )

        log(
            f"Step {step_count} | "
            f"Queue: {step_queue} | "
            f"Waiting: {step_waiting_time:.2f} | "
            f"Arrived: {step_arrived} | "
            f"Active: {active_vehicles} | "
            f"Remaining: {remaining_vehicles}"
        )

        # ------------------------------------------------------------
        # BLOCK 6.6: Stop as soon as traffic has actually finished
        # ------------------------------------------------------------
        # getMinExpectedNumber() == 0 means:
        # - no active vehicles left in the network
        # - no vehicles still expected to depart
        #
        # We also require has_seen_vehicle so we do not terminate before
        # traffic has even started.
        if has_seen_vehicle and remaining_vehicles == 0:
            log("")
            log(f"Traffic finished at simulation step {step_count}. Stopping simulation.")
            break

    average_queue = queue_sum / step_count if step_count > 0 else 0.0
    average_wait_per_vehicle = (
        total_waiting_time / vehicles_cleared if vehicles_cleared > 0 else 0.0
    )
    average_co2_per_vehicle = (
        total_co2_emission / vehicles_cleared if vehicles_cleared > 0 else 0.0
    )
    average_ns_queue = (
        sum(row[2] for row in directional_queue_rows) / len(directional_queue_rows)
        if directional_queue_rows
        else 0.0
    )
    average_we_queue = (
        sum(row[3] for row in directional_queue_rows) / len(directional_queue_rows)
        if directional_queue_rows
        else 0.0
    )
    remaining_vehicles = traci.simulation.getMinExpectedNumber()
    overtime_used = step_count > 1200
    cleared_all = remaining_vehicles == 0

    traci.close()

    queue_csv_path = os.path.splitext(log_file)[0] + "_directional_queue.csv"
    queue_plot_path = os.path.splitext(log_file)[0] + "_directional_queue.png"
    write_directional_queue_outputs(queue_csv_path, queue_plot_path, directional_queue_rows)

    log("")
    log("========== TEST SUMMARY ==========")
    log(f"Route file           : {os.path.basename(ROUTE_FILE)}")
    log(f"Total reward         : {total_reward:.8f}")
    log(f"Average queue        : {average_queue:.4f}")
    log(f"Average NS queue raw : {average_ns_queue:.4f}")
    log(f"Average WE queue raw : {average_we_queue:.4f}")
    log(f"Total waiting time   : {total_waiting_time:.4f}")
    log(f"Total CO2 emission   : {total_co2_emission:.4f} mg")
    log(f"Vehicles cleared     : {vehicles_cleared}")
    log(f"Vehicles left        : {remaining_vehicles}")
    log(f"Overtime used        : {'Yes' if overtime_used else 'No'}")
    log(f"Cleared all vehicles : {'Yes' if cleared_all else 'No'}")
    log(f"Average wait/vehicle : {average_wait_per_vehicle:.4f}")
    log(f"Average CO2/vehicle  : {average_co2_per_vehicle:.4f} mg")
    log(f"Simulation steps     : {step_count}")
    log(f"Directional queue CSV: {queue_csv_path}")
    log(f"Directional queue PNG: {queue_plot_path}")
    log(f"Log file             : {log_file}")
    log("==================================")


# ============================================================
# BLOCK 7: Main entry point
# ============================================================
if __name__ == "__main__":
    run_fixed_simulation()
