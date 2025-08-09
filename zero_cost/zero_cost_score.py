import torch
import numpy as np
from torch.nn.utils import parameters_to_vector

def compute_naswot_score(model, input_shape=(1, 50)):
    """
    Compute NAS-WOT (Neural Architecture Search Without Training) score.
    Uses ReLU activations to estimate architecture quality.
    """
    model.eval()
    hooks = []
    activations = []

    def hook_fn(module, input, output):
        if isinstance(output, torch.Tensor):
            activations.append(output.detach().cpu().numpy().flatten())

    # Register hooks for all ReLU layers
    for module in model.modules():
        if isinstance(module, torch.nn.ReLU):
            hooks.append(module.register_forward_hook(hook_fn))

    # Forward pass with dummy input
    with torch.no_grad():
        dummy_input = torch.randn(*input_shape)
        model(dummy_input)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    if len(activations) == 0:
        return 0.0

    # Compute score based on activation correlations
    act_matrix = np.vstack(activations)
    try:
        gram_matrix = act_matrix @ act_matrix.T
        eigvals = np.linalg.eigvalsh(gram_matrix)
        score = np.sum(np.log(np.abs(eigvals) + 1e-5))
    except:
        score = 0.0
    return score


def satisfies_constraints(model, max_params=50000):
    """Check if model satisfies parameter count constraints."""
    total_params = sum(p.numel() for p in model.parameters())
    return total_params <= max_params


def randomly_sample_architectures(search_space, constraints, score_threshold=0.0, max_samples=10):
    """Sample architectures from search space based on constraints and scores."""
    sampled = []
    np.random.shuffle(search_space)
    
    for model in search_space:
        if satisfies_constraints(model, **constraints):
            score = compute_naswot_score(model)
            if score >= score_threshold:
                sampled.append((model, score))
        if len(sampled) >= max_samples:
            break
    
    return sampled
