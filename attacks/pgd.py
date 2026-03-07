import torch

def pgd_attack(data, model, epsilon, alpha, num_iter, criterion, bounds=None):
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