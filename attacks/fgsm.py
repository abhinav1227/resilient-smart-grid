'''The Problem: Absolute vs. Proportional Scaling
    The Logic: In standard image-based ML, pixel values are normalized between $0$ and $1$. 
            If you apply an FGSM attack with $\epsilon = 0.15$, you are changing the pixel by 15% of its total possible range.
            However, in our power grid, we are passing raw physical values (Megawatts and Megavars) into the network.
            A substation might have a load of $50.0 \text{ MW}$. 
            When the original fgsm_attack calculates data.x + epsilon * grad_sign, it is adding exactly $0.15 \text{ MW}$ to a $50.0 \text{ MW}$ load.
            That is a $0.3\%$ change. The grid physics (and your highly accurate GCN) easily absorbs a fraction of a Megawatt without triggering any instability alarms. 
            The attack is mathematically failing because it is too weak to matter in the physical world!

    The Solution: Proportional FGSMTo fix this, we need to upgrade our attack from an absolute step to a proportional step. 
            We want the attacker to change the load by a percentage of the current load, pushing the limits of our feature_bounds.
            Here is the updated mathematical formulation we will use:
                    $$x_{adv} = x + \epsilon \cdot |x| \cdot \text{sign}(\nabla_x L)$$
                    
            This means if $\epsilon = 0.3$, the attacker will aggressively alter the load by exactly 30% in the most damaging direction.'''

import torch

import torch

def constrained_fgsm_attack(data, model, epsilon, criterion, bounds=None):
    """
    Dynamic Fast Gradient Sign Method.
    Scales the perturbation relative to the overall operational scale of the grid,
    preventing both the "zero-load" trap and the need for hardcoded values.
    """
    data = data.clone()
    data.x.requires_grad = True
    
    # Forward pass
    out = model(data).squeeze(-1)
    
    # Calculate loss to maximize error
    loss = criterion(out, data.y)
    
    model.zero_grad()
    loss.backward()
    
    grad_sign = data.x.grad.sign()
    
    # DYNAMIC GRID SCALING: Calculate the mean absolute power across all nodes in the grid.
    # We add 1e-3 to prevent a division-by-zero error in the impossible case of a completely dead grid.
    grid_scale = data.x.abs().mean() + 1e-3
    
    # Epsilon is now a percentage of the GRID'S AVERAGE load, not the individual node's load.
    step = epsilon * grid_scale * grad_sign
    perturbed_x = data.x + step
    
    # Apply relative physical bounds based on the same grid scale
    if bounds is not None:
        min_multiplier, max_multiplier = bounds
        # The attacker can only shift the load by a multiple of the grid's average scale
        lower_bound = data.x + (min_multiplier * grid_scale)
        upper_bound = data.x + (max_multiplier * grid_scale)
        
        perturbed_x = torch.max(torch.min(perturbed_x, upper_bound), lower_bound)
        
        # Grid physics constraint: Load generally shouldn't drop below 0
        perturbed_x = torch.clamp(perturbed_x, min=0.0)
        
    data.x = perturbed_x.detach()
    
    return data

# proportional scaling
def constrained_fgsm_attack1(data, model, epsilon, criterion, bounds=None):
    """
    Proportional Fast Gradient Sign Method.
    Scales the perturbation relative to the physical magnitude of the node features.
    """
    data = data.clone()
    data.x.requires_grad = True
    
    # Forward pass
    out = model(data).squeeze(-1)
    
    # Calculate loss to maximize error
    loss = criterion(out, data.y)
    
    model.zero_grad()
    loss.backward()
    
    # Collect the sign of the data gradient
    grad_sign = data.x.grad.sign()
    
    # PROPORTIONAL PERTURBATION: 
    # Scale epsilon by the magnitude of the features so we change MW/MVAR by a percentage
    step = epsilon * data.x.abs() * grad_sign
    perturbed_x = data.x + step
    
    # Apply physical bounds (Clipping) to ensure realism
    if bounds is not None:
        min_val, max_val = bounds
        lower_bound = data.x * (1 + min_val)
        upper_bound = data.x * (1 + max_val)
        perturbed_x = torch.max(torch.min(perturbed_x, upper_bound), lower_bound)
        
    data.x = perturbed_x.detach()
    
    return data

def constrained_fgsm_attack0(data, model, epsilon, criterion, bounds=None):
    """
    Fast Gradient Sign Method with physical domain constraints.
    """
    data = data.clone()
    data.x.requires_grad = True
    
    # Forward pass
    out = model(data).squeeze(-1)
    
    # Calculate loss to maximize error (simulate grid instability)
    loss = criterion(out, data.y)
    
    model.zero_grad()
    loss.backward()
    
    # Collect the sign of the data gradient
    grad_sign = data.x.grad.sign()
    
    # Apply the perturbation
    perturbed_x = data.x + epsilon * grad_sign
    
    # Apply physical bounds (Clipping) to ensure realism
    if bounds is not None:
        min_val, max_val = bounds
        # We assume the base features are around a certain scale, 
        # we clamp the perturbation relative to the original features.
        lower_bound = data.x * (1 + min_val)
        upper_bound = data.x * (1 + max_val)
        perturbed_x = torch.max(torch.min(perturbed_x, upper_bound), lower_bound)
        
    data.x = perturbed_x.detach()
    
    return data