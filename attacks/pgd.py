import torch
import random

def pgd_attack(data, model, epsilon, alpha, num_iter, criterion, bounds=None, target_node=None):
    """
    Logic: Spatio-Temporal Targeted Sub-Graph Attack (APT Simulation).
    Reasoning: Targets one node at the current timestep (t=0) to simulate a local breach.
    """
    original_x = data.x.clone().detach()
    perturbed_data = data.clone()
    num_nodes = original_x.shape[0]
    
    if target_node is None:
        target_node = random.randint(0, num_nodes - 1)
        
    # Mask: Only allow perturbation on the targeted node at the latest timestep (-1)
    mask = torch.zeros_like(original_x)
    mask[target_node, -1, :] = 1.0
    
    grid_scale = original_x.abs().mean() + 1e-3
    max_perturbation = epsilon * grid_scale
    
    for _ in range(num_iter):
        perturbed_data.x.requires_grad = True
        out = model(perturbed_data)
        loss = criterion(out, perturbed_data.y)
        
        model.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            step = alpha * grid_scale * perturbed_data.x.grad.sign() * mask
            adv_x = torch.clamp(perturbed_data.x + step, original_x - max_perturbation, original_x + max_perturbation)
            
            if bounds:
                adv_x = torch.clamp(adv_x, min=0.0) # Physical constraint: No negative load
            
            perturbed_data.x = adv_x.detach()

    return perturbed_data