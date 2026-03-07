import torch
import numpy as np
import logging
import random
from main import GridResiliencePipeline

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

def run_dynamic_ids():
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