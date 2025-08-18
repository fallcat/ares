import itertools
import torch
import math

def tree_stability_rate_sample(
    entailment_model, 
    stab_inputs,
    children=None,
    parents=None,
    p = 0.5,
    p_derived = None,
    epsilon=0.1,
    delta=0.1,
    entailment_mode: str = 'granular',
    threshold: float = 0.6,
    temperature: float = 0.0
):
    if p_derived is None:
        p_derived = p

    m = len(stab_inputs)
    n = len(stab_inputs[0]['premises'])
    N = math.ceil(math.log(2*m/delta) / (2 * (epsilon**2)))
    print(f"N: {N}")

    y_pertbs_all = []
    stab_rate_all = []

    samples = torch.zeros(N, n+m)
    for i in range(n+m):
        # sample Y_i which is sampled from bernoulli(p) for N samples
        p_i = p if i < n else p_derived
        trust_pertbs = torch.bernoulli(torch.ones(N) * p_i)
        if i < n:
            samples[:, i] = trust_pertbs
        else:
            premises = stab_inputs[i-n]['premises']
            hypothesis = stab_inputs[i-n]['hypothesis']
            s_pertbs = samples[:, :n]
            # only compute f() for trust_pertbs = 1 so we don't waste compute
            # Only compute entailment for trust_pertbs = 1 to avoid wasting compute
            mask = trust_pertbs == 1
            if mask.any():
                # Get indices where trust_pertbs is 1
                indices = mask.nonzero().squeeze(-1)
                # Only compute for those indices
                filtered_s_pertbs = s_pertbs[indices]
                y_pertbs_filtered = entailment_model(
                    [premises] * len(indices),
                    [hypothesis] * len(indices),
                    s=filtered_s_pertbs,
                    mode=entailment_mode,
                    temperature=temperature
                )
                # Assign results back to the correct positions
                # samples[indices, i] = (y_pertbs_filtered > threshold).float()
                
                # Create full array with zeros for trust_pertbs = 0
                full_y_pertbs = torch.zeros(N).float()
                full_y_pertbs[indices] = y_pertbs_filtered
                samples[indices, i] = (full_y_pertbs >= threshold).float()
                y_pertbs_all.append(full_y_pertbs.tolist())
                stab_rate_all.append((full_y_pertbs >= threshold).float().mean().item())
            else:
                # If no trust_pertbs is 1, append zeros
                y_pertbs_all.append(torch.zeros(N).tolist())
                stab_rate_all.append(0.0)
            # For trust_pertbs = 0, leave as zeros (already initialized as zeros)

    return {
        'inputs': stab_inputs,
        'samples': samples.tolist(),
        'y_pertbs': y_pertbs_all,
        'stability_rates': stab_rate_all,
    }
