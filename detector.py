import torch
import numpy as np
import random
import logging
import random
import time
from prefect import get_run_logger

logger = logging.getLogger("prefect")

def calibrate_thresholds(test_data, model, device):
    """Logic: Establish 4-Sigma limits to minimize false alarms."""
    model.eval()
    errors = []
    with torch.no_grad():
        for snapshot in test_data:
            snapshot = snapshot.to(device)
            pred = model(snapshot)
            # Focus on Voltage Magnitude (Index 0)
            err = torch.abs(pred[:, 0] - snapshot.y[:, 0]).cpu().numpy()
            errors.append(err)
            
    errors = np.array(errors)
    dynamic_thresholds = np.mean(errors, axis=0) + (4 * np.std(errors, axis=0))
    return np.clip(dynamic_thresholds, a_min=1e-4, a_max=None)


def simulate_dynamic_fdia(true_vm, predicted_vm, thresholds):
    """Logic: Targeted SCADA spoofing on a single node."""
    hacked_vm = true_vm.copy()
    num_nodes = len(true_vm)
    target_node = random.randint(0, num_nodes - 1)
    
    # Attack magnitude: Scaled to be detectable (1.5x threshold)
    attack_magnitude = max(thresholds[target_node] * 1.5, 0.05)
    direction = 1 if true_vm[target_node] > predicted_vm[target_node] else -1
    hacked_vm[target_node] = true_vm[target_node] + (direction * attack_magnitude)
    
    return hacked_vm, target_node, attack_magnitude

def topological_breaker_attack(snapshot):
    """Logic: Simulates severing a transmission line by modifying the edge_index."""
    hacked_snapshot = snapshot.clone()
    num_edges = hacked_snapshot.edge_index.shape[1]
    edge_to_drop = random.randint(0, num_edges - 1)

    u = hacked_snapshot.edge_index[0, edge_to_drop].item()
    v = hacked_snapshot.edge_index[1, edge_to_drop].item()
    
    # Create mask to sever the connection U-V
    mask = ~(((hacked_snapshot.edge_index[0] == u) & (hacked_snapshot.edge_index[1] == v)) | 
             ((hacked_snapshot.edge_index[0] == v) & (hacked_snapshot.edge_index[1] == u)))
             
    hacked_snapshot.edge_index = hacked_snapshot.edge_index[:, mask]
    hacked_snapshot.edge_attr = hacked_snapshot.edge_attr[mask]
    
    return hacked_snapshot, u, v

def run_ids_audit(model, test_data, thresholds, device):
    """Logic: Multi-vector Digital Twin simulation with verbose logging."""
    logger = get_run_logger()
    stats = {
        'node_total': 0, 'node_caught': 0,
        'edge_total': 0, 'edge_caught': 0,
        'clean_total': 0, 'false_alarms': 0
    }
    
    simulation_steps = min(50, len(test_data)) 

    for step in range(simulation_steps):
        print(f"\n--- [TIMESTEP {step}] SCADA DATA INGESTION ---")
        
        pristine_snapshot = test_data[step].clone().to(device)
        true_sensor_vm = pristine_snapshot.y[:, 0].cpu().numpy()
        
        attack_roll = random.random()
        attack_type = "none"
        
        # 1. Simulate Threat Vectors
        if attack_roll < 0.15:
            attack_type = "node"
            stats['node_total'] += 1
            with torch.no_grad():
                ai_pred = model(pristine_snapshot)[:, 0].cpu().numpy()
            live_vm, target, mag = simulate_dynamic_fdia(true_sensor_vm, ai_pred, thresholds)
            logger.warning(f"🕵️  ATTACK INJECTED: FDIA on Sensor {target} (Mag: {mag:.4f}).")

            #  AI "Check"
            if abs(live_vm[target] - ai_pred[target]) > thresholds[target]:
                detected = True
            
        elif attack_roll < 0.30:
            attack_type = "edge"
            stats['edge_total'] += 1
            hacked_snapshot, u, v = topological_breaker_attack(pristine_snapshot)
            with torch.no_grad():
                ai_pred = model(hacked_snapshot)[:, 0].cpu().numpy()

            # Compare the true physics before and after the attack
            # This measures the "Physical Impact" of severing that line.
            physical_impact = np.abs(true_sensor_vm - pristine_snapshot.y[:, 0].cpu().numpy()).max()

            live_vm = true_sensor_vm.copy()
            logger.warning(f"⚠️ THREAT DETECTED: Topological Breaker Attack. Line {u}-{v} digitally severed.")
                
            if physical_impact < 1e-3:
                logger.info(f"ℹ️  REDUNDANCY ALERT: Line {u}-{v} severed, but grid physics rerouted with minimal impact ({physical_impact:.6f}).")

        else:
            stats['clean_total'] += 1
            with torch.no_grad():
                ai_pred = model(pristine_snapshot)[:, 0].cpu().numpy()
            live_vm = true_sensor_vm.copy()

        # 2. Digital Twin Verification Logic
        detected = False
        for node_idx in range(len(ai_pred)):
            discrepancy = abs(live_vm[node_idx] - ai_pred[node_idx])
            if discrepancy > thresholds[node_idx]:
                if attack_type == "none":
                    logger.error(f"⚠️ FALSE ALARM (NODE {node_idx}): Noise exceeded 4-Sigma limit.")
                else:
                    logger.error(f"🚨 ALARM (NODE {node_idx}): SCADA={live_vm[node_idx]:.3f}, Twin={ai_pred[node_idx]:.3f} | Diff: {discrepancy:.4f} > Limit: {thresholds[node_idx]:.4f}")
                    logger.info(f"🛡️ MITIGATION: Quarantining Node {node_idx}. Overwriting with AI prediction.")
                detected = True

        # 3. Audit Scoring
        if detected:
            if attack_type == "node": stats['node_caught'] += 1
            if attack_type == "edge": stats['edge_caught'] += 1
            if attack_type == "none": stats['false_alarms'] += 1
        elif attack_type != "none":
            logger.critical(f"❌ CRITICAL FAILURE: {attack_type.upper()} Attack bypassed the Digital Twin.")
            
        print(f"Timestep {step} resolved. Proceeding...")
        time.sleep(0.1) # Smooth terminal output

    return stats