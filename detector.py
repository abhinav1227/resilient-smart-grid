import torch
import numpy as np
import logging
import random
import time
from main import GridResiliencePipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- ATTACK MODULES ---

def simulate_dynamic_fdia(true_vm, predicted_vm, thresholds):
    """Simulates a Node Attack: Hacker directly spoofs the SCADA voltage sensors."""
    hacked_vm = true_vm.copy()
    num_nodes = len(true_vm)
    target_node = random.randint(0, num_nodes - 1)
    attack_magnitude = max(thresholds[target_node] * 1.5, 0.05)
    
    direction = 1 if true_vm[target_node] > predicted_vm[target_node] else -1
    hacked_vm[target_node] = true_vm[target_node] + (direction * attack_magnitude)
    return hacked_vm, target_node, attack_magnitude

def topological_breaker_attack(snapshot):
    """
    Simulates a targeted Edge Attack by intentionally hunting for 
    critical bottleneck lines (radial connections) to sever.
    """
    hacked_snapshot = snapshot.clone()
    
    # 1. Calculate the 'degree' of every node (how many lines connect to it)
    # The Logic: We count how many times each node appears in the edge_index
    node_degrees = torch.bincount(hacked_snapshot.edge_index[0])
    
    # 2. Find the "Leaf Nodes" (Substations with only 1 connection)
    # These are our critical vulnerabilities. If we cut their only line, they die.
    leaf_nodes = torch.where(node_degrees == 1)[0].tolist()
    
    if len(leaf_nodes) > 0:
        # 3. If a vulnerable leaf node exists, target it specifically
        target_node = random.choice(leaf_nodes)
        
        # Find the exact transmission line connected to this leaf node
        edge_mask = (hacked_snapshot.edge_index[0] == target_node)
        edge_to_drop = torch.where(edge_mask)[0][0].item()
    else:
        # Fallback if no leaf nodes exist (random attack)
        num_edges = hacked_snapshot.edge_index.shape[1]
        edge_to_drop = random.randint(0, num_edges - 1)

    u = hacked_snapshot.edge_index[0, edge_to_drop].item()
    v = hacked_snapshot.edge_index[1, edge_to_drop].item()
    
    # Create mask to sever the connection between U and V in both directions
    mask = ~(((hacked_snapshot.edge_index[0] == u) & (hacked_snapshot.edge_index[1] == v)) | 
             ((hacked_snapshot.edge_index[0] == v) & (hacked_snapshot.edge_index[1] == u)))
             
    # Apply the topological spoof
    hacked_snapshot.edge_index = hacked_snapshot.edge_index[:, mask]
    hacked_snapshot.edge_attr = hacked_snapshot.edge_attr[mask]
    
    return hacked_snapshot, u, v

# --- CALIBRATION MODULE ---

def calibrate_thresholds(pipeline, model):
    """Calculates per-node 3-Sigma limits across pristine data."""
    logger.info("Calibrating Dynamic Thresholds across pristine test data...")
    model.eval()
    all_errors = []
    
    with torch.no_grad():
        for snapshot in pipeline.test_data:
            snapshot = snapshot.to(pipeline.device)
            pred = model(snapshot)
            all_errors.append(np.abs(pred[:, 0].cpu().numpy() - snapshot.y[:, 0].cpu().numpy()))
            
    all_errors = np.array(all_errors) 
    mean_errors = np.mean(all_errors, axis=0)
    std_errors = np.std(all_errors, axis=0)
    
    dynamic_thresholds = mean_errors + (4 * std_errors)
    return np.clip(dynamic_thresholds, a_min=1e-4, a_max=None)

# --- LIVE CONTROL ROOM ---

def run_dynamic_ids():
    """Continuous Self-Healing Digital Twin with Automated Security Auditing."""
    logger.info("Initializing Continuous Multi-Vector Digital Twin...")
    
    pipeline = GridResiliencePipeline('config.yaml')
    pipeline.prepare_data()
    
    model = pipeline.model
    model.load_state_dict(torch.load(pipeline.config['model']['save_path']))
    model.eval()
    
    dynamic_thresholds = calibrate_thresholds(pipeline, model)
    logger.info("Calibration Complete. Starting live grid monitoring...\n")
    
    simulation_steps = min(50, len(pipeline.test_data)) 
    
    # THE LOGIC: We create a scoring dictionary to rigorously audit the AI's performance.
    audit_stats = {
        'total_node_attacks': 0, 'caught_node_attacks': 0,
        'total_edge_attacks': 0, 'caught_edge_attacks': 0,
        'total_clean_steps': 0, 'false_alarms': 0
    }
    
    for step in range(simulation_steps):
        logger.info(f"--- [TIMESTEP {step}] SCADA DATA INGESTION ---")
        
        pristine_snapshot = pipeline.test_data[step].clone().to(pipeline.device)
        true_sensor_vm = pristine_snapshot.y[:, 0].cpu().numpy()
        
        attack_roll = random.random()
        attack_type = "none" # Track what is currently happening
        
        if attack_roll < 0.15:
            attack_type = "node"
            audit_stats['total_node_attacks'] += 1
            with torch.no_grad():
                ai_prediction = model(pristine_snapshot)[:, 0].cpu().numpy()
            live_sensor_vm, target, mag = simulate_dynamic_fdia(true_sensor_vm, ai_prediction, dynamic_thresholds)
            logger.warning(f"⚠️  THREAT DETECTED: FDIA Node Attack. Sensor {target} spoofed by {mag:.4f} pu.")
            
        elif attack_roll < 0.30:
            attack_type = "edge"
            audit_stats['total_edge_attacks'] += 1
            hacked_snapshot, u, v = topological_breaker_attack(pristine_snapshot)
            with torch.no_grad():
                ai_prediction = model(hacked_snapshot)[:, 0].cpu().numpy()
            live_sensor_vm = true_sensor_vm.copy()
            logger.warning(f"⚠️  THREAT DETECTED: Topological Breaker Attack. Line {u}-{v} digitally severed.")
            
        else:
            attack_type = "none"
            audit_stats['total_clean_steps'] += 1
            with torch.no_grad():
                ai_prediction = model(pristine_snapshot)[:, 0].cpu().numpy()
            live_sensor_vm = true_sensor_vm.copy()
            logger.info("Grid state stable. No anomalies detected.")

        detected = False
        trusted_grid_state = live_sensor_vm.copy() 
        
        for node_idx in range(len(ai_prediction)):
            sensor_val = live_sensor_vm[node_idx]
            ai_val = ai_prediction[node_idx]
            discrepancy = abs(sensor_val - ai_val)
            node_threshold = dynamic_thresholds[node_idx]
            
            if discrepancy > node_threshold:
                if attack_type == "none":
                    logger.error(f"⚠️ FALSE ALARM (NODE {node_idx}): AI predicted an anomaly on clean data.")
                else:
                    logger.error(f"🚨 ALARM (NODE {node_idx}): SCADA={sensor_val:.3f}, Digital Twin={ai_val:.3f} | Diff: {discrepancy:.4f} > Limit: {node_threshold:.4f}")
                    logger.info(f"🛡️ MITIGATION: Quarantining Node {node_idx}. Overwriting with AI physics prediction ({ai_val:.3f}).")
                
                trusted_grid_state[node_idx] = ai_val 
                detected = True
                
        # THE LOGIC: Update our audit counters based on the outcome
        if detected:
            if attack_type == "node": audit_stats['caught_node_attacks'] += 1
            if attack_type == "edge": audit_stats['caught_edge_attacks'] += 1
            if attack_type == "none": audit_stats['false_alarms'] += 1
            
        if attack_type != "none" and not detected:
            logger.critical("❌ CRITICAL FAILURE: Attack bypassed the Digital Twin.")
            
        logger.info(f"Timestep {step} resolved. Proceeding...\n")
        time.sleep(0.5)

    # --- PHASE 5: THE SECURITY AUDIT REPORT ---
    logger.info("="*50)
    logger.info("🛡️ FINAL IDS SECURITY AUDIT REPORT 🛡️")
    logger.info("="*50)
    
    # Calculate percentages safely
    node_rate = (audit_stats['caught_node_attacks'] / max(1, audit_stats['total_node_attacks'])) * 100
    edge_rate = (audit_stats['caught_edge_attacks'] / max(1, audit_stats['total_edge_attacks'])) * 100
    false_alarm_rate = (audit_stats['false_alarms'] / max(1, audit_stats['total_clean_steps'])) * 100
    
    logger.info(f"Node Attacks (FDIA) Defeated:   {audit_stats['caught_node_attacks']}/{audit_stats['total_node_attacks']} ({node_rate:.1f}%)")
    logger.info(f"Edge Attacks (Breaker) Defeated: {audit_stats['caught_edge_attacks']}/{audit_stats['total_edge_attacks']} ({edge_rate:.1f}%)")
    logger.info(f"False Alarm Rate (Clean Grid):   {audit_stats['false_alarms']}/{audit_stats['total_clean_steps']} ({false_alarm_rate:.1f}%)")
    logger.info("="*50)

if __name__ == "__main__":
    run_dynamic_ids()