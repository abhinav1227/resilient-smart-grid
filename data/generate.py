import os
import numpy as np
import pandapower as pp
import logging

logger = logging.getLogger(__name__)

def generate_time_series(net, n_timesteps=500, seed=42, save_path=None):
    """
    Generates dynamic load profiles and solves power flow for any given grid topology.
    """
    if save_path and os.path.exists(save_path):
        logger.info(f"Loading existing data from {save_path}")
        data = np.load(save_path)
        return data['features'], data['targets']

    np.random.seed(seed)
    features, targets = [], []
    num_buses = len(net.bus)
    
    logger.info(f"Simulating power flow for {num_buses} buses over {n_timesteps} timesteps...")
    
    successful_steps = 0
    for i in range(n_timesteps):
        # Add realistic Gaussian noise to active and reactive power
        net.load['p_mw'] = net.load['p_mw'] * (1 + 0.1 * np.random.randn(len(net.load)))
        net.load['q_mvar'] = net.load['q_mvar'] * (1 + 0.1 * np.random.randn(len(net.load)))
        
        try:
            pp.runpp(net, numba=False)
            successful_steps += 1
        except pp.pandapower.powerflow.LoadflowNotConverged:
            # Professionally log non-convergence instead of silently failing
            logger.debug(f"Power flow failed to converge at timestep {i}. Skipping.")
            continue
            
        bus_load_p = np.zeros(num_buses)
        bus_load_q = np.zeros(num_buses)
        
        for _, load in net.load.iterrows():
            bus_load_p[int(load.bus)] += load.p_mw
            bus_load_q[int(load.bus)] += load.q_mvar
            
        feat = np.column_stack([bus_load_p, bus_load_q])

        # EXTRACT BOTH MAGNITUDE AND ANGLE
        v_mag = net.res_bus.vm_pu.values
        v_ang = net.res_bus.va_degree.values * (np.pi / 180.0) # Convert degrees to radians for better numerical stability in ML models
        
        # Stack them to create a 2D target matrix for each timestep
        target = np.column_stack([v_mag, v_ang])

        features.append(feat)
        targets.append(target)

    features = np.array(features, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(save_path, features=features, targets=targets)
        logger.info(f"Successfully generated {successful_steps} samples. Saved to {save_path}")

    return features, targets