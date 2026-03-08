import os
import numpy as np
import pandapower as pp
import logging

logger = logging.getLogger(__name__)

def generate_time_series(net, n_timesteps=500, seed=42, save_path=None):
    """
    Generates dynamic load profiles and extracts static physical edge attributes
    (Resistance and Reactance) for physics-informed neural networks.
    """
    if save_path and os.path.exists(save_path):
        logger.info(f"Loading existing AC physics data from {save_path}")
        data = np.load(save_path)
        return data['features'], data['targets'], data['edge_index'], data['edge_attr']

    np.random.seed(seed)
    features, targets = [], []
    num_buses = len(net.bus)
    
    logger.info(f"Extracting static physical topology and impedance parameters...")
    
    # --- UPGRADE: EXTRACT STATIC PHYSICS (EDGE ATTRIBUTES) ---
    # 1. Get the topological connections
    from_buses = net.line.from_bus.values
    to_buses = net.line.to_bus.values
    
    # 2. Calculate the total physical Resistance (R) and Reactance (X)
    r_total = net.line.r_ohm_per_km.values * net.line.length_km.values
    x_total = net.line.x_ohm_per_km.values * net.line.length_km.values
    
    # 3. Create forward directed edges
    edge_index_forward = np.vstack([from_buses, to_buses])
    edge_attr_forward = np.column_stack([r_total, x_total])
    
    # 4. AC power flows both ways, so we duplicate for backward edges
    edge_index_backward = np.vstack([to_buses, from_buses])
    edge_attr_backward = edge_attr_forward.copy() 
    
    # 5. Combine into final undirected graph arrays
    edge_index = np.hstack([edge_index_forward, edge_index_backward])
    edge_attr = np.vstack([edge_attr_forward, edge_attr_backward])
    # ---------------------------------------------------------

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
        v_ang = net.res_bus.va_degree.values * (np.pi / 180.0) 
        
        target = np.column_stack([v_mag, v_ang])

        features.append(feat)
        targets.append(target)

    features = np.array(features, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # UPGRADE: Save the new edge arrays into the .npz archive
        np.savez(save_path, features=features, targets=targets, edge_index=edge_index, edge_attr=edge_attr)
        logger.info(f"Successfully generated {successful_steps} samples with physics edges. Saved to {save_path}")

    return features, targets, edge_index, edge_attr