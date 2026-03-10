import numpy as np
import pandapower as pp
import torch

def generate_time_series(net, n_timesteps=500, seed=42):
    """
    Logic: Pure physics simulation. 
    Reasoning: Returns raw numpy/tensor objects so the pipeline can manage persistence.
    """
    np.random.seed(seed)
    features, targets = [], []
    num_buses = len(net.bus)
    
    # 1. Extract Static Topology
    from_buses = net.line.from_bus.values
    to_buses = net.line.to_bus.values
    r_total = net.line.r_ohm_per_km.values * net.line.length_km.values
    x_total = net.line.x_ohm_per_km.values * net.line.length_km.values
    
    edge_index = np.hstack([np.vstack([from_buses, to_buses]), np.vstack([to_buses, from_buses])])
    edge_attr = np.vstack([np.column_stack([r_total, x_total]), np.column_stack([r_total, x_total])])

    # 2. Simulation Loop
    for _ in range(n_timesteps):
        net.load['p_mw'] *= (1 + 0.1 * np.random.randn(len(net.load)))
        net.load['q_mvar'] *= (1 + 0.1 * np.random.randn(len(net.load)))
        
        try:
            pp.runpp(net, numba=False)
            bus_load_p = np.zeros(num_buses)
            bus_load_q = np.zeros(num_buses)
            for _, load in net.load.iterrows():
                bus_load_p[int(load.bus)] += load.p_mw
                bus_load_q[int(load.bus)] += load.q_mvar
            
            features.append(np.column_stack([bus_load_p, bus_load_q]))
            targets.append(np.column_stack([net.res_bus.vm_pu.values, net.res_bus.va_degree.values * (np.pi / 180.0)]))
        except pp.LoadflowNotConverged:
            continue

    return np.array(features, dtype=np.float32), np.array(targets, dtype=np.float32), edge_index, edge_attr