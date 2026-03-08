import torch

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import random

def pgd_attack(data, model, epsilon, alpha, num_iter, criterion, bounds=None, target_node=None):
    """
    Version 3: Spatio-Temporal Targeted Sub-Graph Attack.
    Simulates a highly realistic APT (Advanced Persistent Threat) by 
    surgically targeting a single node at the live timestep.
    """
    # Keep the original data safe to project against later
    original_x = data.x.clone().detach()
    perturbed_data = data.clone()
    
    # --- THE SPATIO-TEMPORAL APT MASK LOGIC ---
    # 1. Determine the shape and size of the grid
    is_temporal = len(original_x.shape) == 3
    num_nodes = original_x.shape[0]
    
    # 2. Dynamic Target Selection
    # If no specific node is requested, randomly select one to simulate an unpredictable APT
    if target_node is None:
        target_node = random.randint(0, num_nodes - 1)
        
    # 3. Build the Mask
    mask = torch.ones_like(original_x)
    
    if is_temporal:
        # Zero out the entire mask to block history and neighboring node alterations
        mask = torch.zeros_like(original_x)
        # Apply the strike ONLY to the targeted node at the live timestep (index -1)
        mask[target_node, -1, :] = 1.0
    else:
        # Fallback just in case you ever run this on the V1 non-temporal model
        mask = torch.zeros_like(original_x)
        mask[target_node, :] = 1.0
    # ------------------------------------------
    
    # Calculate the dynamic grid scale
    grid_scale = original_x.abs().mean() + 1e-3
    max_perturbation = epsilon * grid_scale
    
    for i in range(num_iter):
        perturbed_data.x = perturbed_data.x.detach().clone()
        perturbed_data.x.requires_grad = True
        
        # Forward pass 
        out = model(perturbed_data)
        loss = criterion(out, perturbed_data.y)
        
        # Calculate gradients
        model.zero_grad()
        loss.backward()
        
        grad_sign = perturbed_data.x.grad.sign()
        
        # Take a SMALL step restricted by the mask
        with torch.no_grad():
            # THE EXECUTION: Multiply the step by the mask. 
            # Gradients for past timesteps AND non-targeted nodes are instantly zeroed out.
            step = alpha * grid_scale * grad_sign * mask
            adv_x = perturbed_data.x + step
            
            # PROJECTION: Ensure the total accumulated change doesn't exceed epsilon
            perturbation = adv_x - original_x
            perturbation = torch.clamp(perturbation, -max_perturbation, max_perturbation)
            adv_x = original_x + perturbation
            
            # Apply physical operational bounds (e.g., no negative load)
            if bounds is not None:
                min_multiplier, max_multiplier = bounds
                lower_bound = original_x + (min_multiplier * grid_scale)
                upper_bound = original_x + (max_multiplier * grid_scale)
                
                adv_x = torch.max(torch.min(adv_x, upper_bound), lower_bound)
                adv_x = torch.clamp(adv_x, min=0.0)
            
            # Update the working data for the next iteration
            perturbed_data.x = adv_x

    return perturbed_data

def pgd_attack2(data, model, epsilon, alpha, num_iter, criterion, bounds=None):
    """
    Version 2.1 (Spatio-Temporal) Projected Gradient Descent Attack.
    Locks the attacker into only perturbing the final timestep (t_0),
    forcing them to deal with the immutable physical inertia of the past.
    """
    # We keep the original data safe to project against later
    original_x = data.x.clone().detach()
    
    # We create a working copy that we will iteratively perturb
    perturbed_data = data.clone()
    
    # --- THE TEMPORAL MASK LOGIC ---
    # We check if the data is 3-dimensional (Nodes, Time, Features)
    is_temporal = len(original_x.shape) == 3
    
    # Create a mask of 1s (allows attacks everywhere by default)
    mask = torch.ones_like(original_x)
    
    if is_temporal:
        # If temporal, zero out the entire mask, blocking all attacks...
        mask = torch.zeros_like(original_x)
        # ...and ONLY allow attacks on the final timestep (index -1)
        mask[:, -1, :] = 1.0
    # -------------------------------
    
    # Calculate the dynamic grid scale
    grid_scale = original_x.abs().mean() + 1e-3
    max_perturbation = epsilon * grid_scale
    
    for i in range(num_iter):
        perturbed_data.x = perturbed_data.x.detach().clone()
        perturbed_data.x.requires_grad = True
        
        # 1. Forward pass (REMOVED .squeeze(-1) for V2 compatibility)
        out = model(perturbed_data)
        loss = criterion(out, perturbed_data.y)
        
        # 2. Calculate gradients
        model.zero_grad()
        loss.backward()
        
        grad_sign = perturbed_data.x.grad.sign()
        
        # 3. Take a SMALL step restricted by the mask
        with torch.no_grad():
            # THE FIX: Multiply the step by the mask. 
            # Gradients for past timesteps are instantly zeroed out.
            step = alpha * grid_scale * grad_sign * mask
            adv_x = perturbed_data.x + step
            
            # 4. PROJECTION: Ensure the total accumulated change doesn't exceed epsilon
            perturbation = adv_x - original_x
            perturbation = torch.clamp(perturbation, -max_perturbation, max_perturbation)
            adv_x = original_x + perturbation
            
            # 5. Apply physical operational bounds (e.g., no negative load)
            if bounds is not None:
                min_multiplier, max_multiplier = bounds
                lower_bound = original_x + (min_multiplier * grid_scale)
                upper_bound = original_x + (max_multiplier * grid_scale)
                
                adv_x = torch.max(torch.min(adv_x, upper_bound), lower_bound)
                adv_x = torch.clamp(adv_x, min=0.0)
            
            # Update the working data for the next iteration
            perturbed_data.x = adv_x

    return perturbed_data

def pgd_attack1(data, model, epsilon, alpha, num_iter, criterion, bounds=None):
    """
    Projected Gradient Descent Attack.
    An iterative, highly optimized attack that dynamically recalculates 
    gradients to find the absolute worst-case physical perturbation.
    """
    # We keep the original data safe to project against later
    original_x = data.x.clone().detach()
    
    # We create a working copy that we will iteratively perturb
    perturbed_data = data.clone()
    perturbed_data.x = perturbed_data.x.detach().clone()
    perturbed_data.x.requires_grad = True
    
    # Calculate the dynamic grid scale just like our advanced FGSM
    grid_scale = original_x.abs().mean() + 1e-3
    max_perturbation = epsilon * grid_scale
    
    for i in range(num_iter):
        perturbed_data.x.requires_grad = True
        
        # 1. Forward pass with the current perturbed state
        out = model(perturbed_data).squeeze(-1)
        loss = criterion(out, perturbed_data.y)
        
        # 2. Calculate gradients
        model.zero_grad()
        loss.backward()
        
        grad_sign = perturbed_data.x.grad.sign()
        
        # 3. Take a SMALL step (alpha) instead of a massive epsilon leap
        with torch.no_grad():
            step = alpha * grid_scale * grad_sign
            adv_x = perturbed_data.x + step
            
            # 4. PROJECTION: Ensure the total accumulated change doesn't exceed our maximum allowed epsilon
            # This prevents the iterative attack from growing infinitely
            perturbation = adv_x - original_x
            perturbation = torch.clamp(perturbation, -max_perturbation, max_perturbation)
            adv_x = original_x + perturbation
            
            # 5. Apply physical operational bounds (e.g., no negative load)
            if bounds is not None:
                min_multiplier, max_multiplier = bounds
                lower_bound = original_x + (min_multiplier * grid_scale)
                upper_bound = original_x + (max_multiplier * grid_scale)
                
                adv_x = torch.max(torch.min(adv_x, upper_bound), lower_bound)
                adv_x = torch.clamp(adv_x, min=0.0)
            
            # Update the working data for the next iteration
            perturbed_data.x = adv_x.clone().detach()

    return perturbed_data