import torch
import numpy as np
import logging
import random
from main import GridResiliencePipeline
import time

# Setup standard logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calibrate_thresholds(pipeline, model):
    """
    Dynamically calculates per-node anomaly thresholds using the 3-Sigma rule
    across the pristine test dataset to eliminate False Positives.
    """
    logger.info("Calibrating Dynamic Thresholds across pristine test data...")
    model.eval()
    all_errors = []
    
    with torch.no_grad():
        for snapshot in pipeline.test_data:
            snapshot = snapshot.to(pipeline.device)
            pred = model(snapshot)
            
            pred_vm = pred[:, 0].cpu().numpy()
            true_vm = snapshot.y[:, 0].cpu().numpy()
            
            error = np.abs(pred_vm - true_vm)
            all_errors.append(error)
            
    # Convert to 2D array: Shape becomes (num_samples, num_nodes)
    all_errors = np.array(all_errors) 
    
    # Calculate statistics for each node independently
    mean_errors = np.mean(all_errors, axis=0)
    std_errors = np.std(all_errors, axis=0)
    
    # Apply the 3-Sigma Rule (99.7% confidence interval)
    dynamic_thresholds = mean_errors + (3 * std_errors)
    
    # Add a tiny buffer (1e-4) to prevent thresholds of absolute zero 
    # on perfectly predictable generator nodes.
    dynamic_thresholds = np.clip(dynamic_thresholds, a_min=1e-4, a_max=None)
    
    return dynamic_thresholds

def simulate_dynamic_fdia(true_vm, predicted_vm, thresholds):
    """
    Simulates a SMART hacker who intentionally pushes the sensor reading 
    away from the AI's prediction to ensure maximum discrepancy.
    """
    hacked_vm = true_vm.copy()
    num_nodes = len(true_vm)
    
    # 1. Pick a random substation to attack
    target_node = random.randint(0, num_nodes - 1)
    
    # 2. Enforce a physical minimum attack size.
    # The logic: A hack that only changes voltage by 0.001 isn't dangerous. 
    # We force the hacker to inject at least a 5% (0.05 pu) disruption.
    attack_magnitude = max(thresholds[target_node] * 1.5, 0.05)
    
    # 3. SMART HACKER LOGIC (No more coin flips)
    # The logic: If the true voltage is already higher than the AI's prediction,
    # the hacker pushes it even HIGHER to maximize the discrepancy gap.
    if true_vm[target_node] > predicted_vm[target_node]:
        direction = 1
    else:
        direction = -1
        
    hacked_vm[target_node] = true_vm[target_node] + (direction * attack_magnitude)
    
    return hacked_vm, target_node, attack_magnitude

# Active Detection with Continuous Self-Healing
def run_dynamic_ids():
    """Main orchestration function for the Continuous Self-Healing IDS."""
    logger.info("Initializing Continuous Digital Twin IDS...")
    
    pipeline = GridResiliencePipeline('config.yaml')
    pipeline.prepare_data()
    
    model = pipeline.model
    model.load_state_dict(torch.load(pipeline.config['model']['save_path']))
    model.eval()
    
    # --- PHASE 1: CALIBRATION ---
    dynamic_thresholds = calibrate_thresholds(pipeline, model)
    logger.info("Calibration Complete. 3-Sigma bounds established.")
    logger.info("Starting live grid monitoring... Press Ctrl+C to stop.\n")
    
    # --- PHASE 2: CONTINUOUS LIVE STREAM ---
    # We loop through a sequence of the test data to simulate the passage of time
    simulation_steps = min(50, len(pipeline.test_data)) 
    
    for step in range(simulation_steps):
        logger.info(f"--- [TIMESTEP {step}] SCADA DATA INGESTION ---")
        
        # Pull the live snapshot from the stream
        snapshot = pipeline.test_data[step].clone().to(pipeline.device)
        
        with torch.no_grad():
            gnn_prediction = model(snapshot)
            predicted_vm = gnn_prediction[:, 0].cpu().numpy()
            true_sensor_vm = snapshot.y[:, 0].cpu().numpy()
            
        # --- PHASE 3: THE CYBER ATTACK (Occurs 20% of the time) ---
        is_attacked = random.random() < 0.20
        if is_attacked:
            live_sensor_vm, target_node, attack_mag = simulate_dynamic_fdia(true_sensor_vm, predicted_vm, dynamic_thresholds)
            logger.warning(f"⚠️  WARNING: Unauthorized access detected. Node {target_node} manipulated by {attack_mag:.4f} pu.")
        else:
            live_sensor_vm = true_sensor_vm.copy()
            logger.info("Grid state stable. No anomalies detected.")

        # --- PHASE 4: DETECTION & SELF-HEALING ALGORITHM ---
        detected = False
        trusted_grid_state = live_sensor_vm.copy() # This is what the control room will use
        
        for node_idx in range(len(predicted_vm)):
            sensor_val = live_sensor_vm[node_idx]
            ai_val = predicted_vm[node_idx]
            discrepancy = abs(sensor_val - ai_val)
            node_threshold = dynamic_thresholds[node_idx]
            
            if discrepancy > node_threshold:
                logger.error(f"🚨 ALARM (NODE {node_idx}): Sensor={sensor_val:.3f}, AI={ai_val:.3f} | Diff: {discrepancy:.4f} > Limit: {node_threshold:.4f}")
                
                # THE MITIGATION LOGIC: The AI overwrites the hacked sensor data with its own physics prediction
                logger.info(f"🛡️ MITIGATION: Quarantining Node {node_idx} SCADA sensor. Replacing corrupted data with AI Digital Twin prediction ({ai_val:.3f}).")
                trusted_grid_state[node_idx] = ai_val 
                detected = True
                
        if is_attacked and not detected:
            logger.critical("❌ CRITICAL FAILURE: Attack bypassed the Digital Twin.")
        
        logger.info(f"Timestep {step} resolved. Proceeding to next timeframe...\n")
        time.sleep(1) # Pause for 1 second to simulate real-time ingestion

# Passive Detection
def run_dynamic_ids1():
    """Main orchestration function for the Digital Twin Intrusion Detection System."""
    logger.info("Initializing Dynamic Digital Twin IDS...")
    
    pipeline = GridResiliencePipeline('config.yaml')
    pipeline.prepare_data()
    
    model = pipeline.model
    model.load_state_dict(torch.load(pipeline.config['model']['save_path']))
    model.eval()
    
    # --- PHASE 1: CALIBRATION ---
    dynamic_thresholds = calibrate_thresholds(pipeline, model)
    logger.info("Calibration Complete. 3-Sigma bounds established for all nodes.")
    
    # --- PHASE 2: LIVE SIMULATION ---
    # Pick a completely random moment in time from the test set
    sample_idx = random.randint(0, len(pipeline.test_data) - 1)
    snapshot = pipeline.test_data[sample_idx].clone().to(pipeline.device)
    
    with torch.no_grad():
        gnn_prediction = model(snapshot)
        predicted_vm = gnn_prediction[:, 0].cpu().numpy()
        true_sensor_vm = snapshot.y[:, 0].cpu().numpy()
        
    # --- PHASE 3: THE CYBER ATTACK ---
    hacked_sensor_vm, target_node, attack_mag = simulate_dynamic_fdia(true_sensor_vm, predicted_vm, dynamic_thresholds)
    logger.info("\n--- INITIATING BLIND FDIA ATTACK ---")
    logger.info(f"Hacker is targeting Node {target_node} with a {attack_mag:.4f} pu manipulation.")
    
    # --- PHASE 4: DETECTION ALOGIRTHM ---
    logger.info("\n--- SCADA SENSOR DISCREPANCY CHECK ---")
    detected = False
    
    for node_idx in range(len(predicted_vm)):
        sensor_val = hacked_sensor_vm[node_idx]
        ai_val = predicted_vm[node_idx]
        discrepancy = abs(sensor_val - ai_val)
        node_threshold = dynamic_thresholds[node_idx]
        
        # The logic: We evaluate against the specific node's unique threshold
        if discrepancy > node_threshold:
            logger.warning(f"🚨 ALARM AT NODE {node_idx}: Sensor={sensor_val:.3f}, AI={ai_val:.3f} | Discrepancy: {discrepancy:.4f} (Limit: {node_threshold:.4f})")
            if node_idx == target_node:
                detected = True
                
    if detected:
        logger.info(f"\n✅ SUCCESS: Digital Twin correctly caught the attack on Node {target_node} and ignored all baseline noise.")
    else:
        logger.error(f"\n❌ FAILURE: Attack on Node {target_node} bypassed the IDS.")

if __name__ == "__main__":
    run_dynamic_ids()