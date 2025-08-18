import itertools
import torch
import math

def sample_s_pertbs(
    s: torch.LongTensor,
    p: float,
    num_samples: int,
    exact: bool = False,
    existing_samples: torch.Tensor = None,
    existing_counts: torch.Tensor = None
):
    original_shape = s.shape
    s = s.view(-1)
    n = s.shape[-1]

    if existing_samples is not None:
        samples = existing_samples
        counts = existing_counts if existing_counts is not None else torch.ones(samples.shape[0], device=samples.device)
        probs = counts / counts.sum()
    else:
        # existing non-exact sampling
        samples = (torch.rand(num_samples, n).to(s.device) <= p) * s
        unique_samples, counts = samples.unique(dim=0, return_counts=True)
        samples = unique_samples
        probs = counts / counts.sum()

    # if len(samples) == 2**n:
    #     exact = True

    if exact:
        # Enumerate all 2^|s| subsets explicitly
        if existing_samples is None:
            s_np = s.cpu().numpy()
            all_combinations = list(itertools.product([0, 1], repeat=n))
            samples = torch.tensor(all_combinations, device=s.device, dtype=torch.long)

            # Compute log probability of each sample explicitly to ensure numerical stability
            log_prob = []
            for comb in samples:
                comb_log_prob = 0.0
                for orig_bit, pertb_bit in zip(s_np, comb.cpu().numpy()):
                    if orig_bit == 1:
                        comb_log_prob += torch.log(torch.tensor(p if pertb_bit == 1 else 1 - p))
                    else:
                        if pertb_bit != 0:
                            comb_log_prob = float('-inf')
                            break
                log_prob.append(comb_log_prob)

            log_prob = torch.tensor(log_prob, device=s.device, dtype=torch.float)
            valid_mask = log_prob != float('-inf')

            samples = samples[valid_mask]
            log_prob = log_prob[valid_mask]
            probs = torch.exp(log_prob - torch.logsumexp(log_prob, dim=0))  # Correct stable normalization

            # Scale counts to ensure minimum count is at least 1
            min_nonzero_prob = probs[probs > 0].min()
            counts = (probs / min_nonzero_prob).ceil().long()

        return samples, counts, probs, exact
    else:
        # Sample according to the distribution in counts
        # if num_samples < len(samples):
        # Sample indices according to probability distribution
        indices = torch.multinomial(probs, num_samples, replacement=True)
        
        # Get the selected samples and their counts
        selected_samples = samples[indices]
        
        # Remove duplicates and count occurrences
        unique_selected_samples, new_counts = selected_samples.unique(dim=0, return_counts=True)
        
        return unique_selected_samples, new_counts, new_counts / new_counts.sum(), exact
        # else:
        #     return samples, counts, probs, exact

def stability_rate_deterministic(
    f,
    premises: list, # this is NOT batched
    hypothesis: str, # this is NOT batched
    p: float, # percent of times corrupted 
    epsilon: float = 0.1, 
    delta: float = 0.1, 
    batch_size: int=16, 
    return_all: bool=False,
    verbose=False,
    samples=None, # allow passing in existing samples, this would do for intermediate claims, whose entailment is dependent on previous claims
    counts=None, # if samples is not None, then if counts is not None, use this counts, or else treat all samples the same (counts be 1 for all).
    exact: bool = False,
    entailment_mode: str = 'granular',
    N: int = None
):
    n = len(premises)
    s = torch.ones(n)
    
    
    # reference prediction
    y = f([premises], [hypothesis], [s], mode=entailment_mode) # can we cache this?
    
    if N is None:
        N = int(math.log(2/delta) / (2 * (epsilon**2))) + 1
    if verbose:
        print('N', N)
    
    all_y_pertbs = []
    
    # need to fix the sampling. when no samples are provided, we should sample from the entire space.
    # if exact is not True, then we should sample N using the probability distribution from counts.
    # when samples are provided, we should sample from the existing samples and still sample N using the probability distribution from counts.
    if samples is None or not exact:
        samples, counts, perc_counts, exact = sample_s_pertbs(s, p, N, exact=exact, existing_samples=samples, existing_counts=counts)
    else:
        if counts is None:
            counts = torch.ones(samples.shape[0]).to(samples.device)
        perc_counts = counts / counts.sum()
        # exact = None
    pbar = torch.split(samples, batch_size)
    if verbose:
        pbar = tqdm(pbar)
    
    for s_pertbs in pbar:
        repeat_pattern = s_pertbs.size(0)
        y_pertbs = f(
            [premises] * repeat_pattern,
            [hypothesis] * repeat_pattern,
            s=s_pertbs,
            mode=entailment_mode
        )

        all_y_pertbs.append(((y_pertbs >= 0.5) == (y >= 0.5)).float()) # this should be checking if entailment is the same
        
    all_y_pertbs = torch.cat(all_y_pertbs, dim=0)

    stab_rate = (all_y_pertbs * perc_counts).sum()

    if return_all:
        return {
            'stability_rate': stab_rate.item(),
            'y': y.tolist(),
            'y_pertbs': all_y_pertbs.tolist(),
            'samples': samples.tolist(),
            'counts': counts.tolist(),
            'perc_counts': perc_counts.tolist(),
            'exact': exact, # bool: True: doing exact all combinations, False: doing sampling
            'entailment_mode': entailment_mode if isinstance(entailment_mode, str) else entailment_mode.__dict__
        }
    else:
        return stab_rate

def tree_stability_rate_deterministic_equal(
    entailment_model, 
    stab_inputs,
    children=None,
    parents=None,
    exact=True, 
    p = 0.5,
    epsilon=0.1,
    delta=0.1,
    entailment_mode: str = 'granular'
):
    stab_rate_results_all_hyps = []

    m = len(stab_inputs)
    n = len(stab_inputs[0]['premises'])
    N = math.ceil(math.log(2*m/delta) / (2 * (epsilon**2))) # for union bound, we need this to make the whole tree to be within epsilon and delta

    samples = None
    counts = None
    perc_counts = None
    for ii in range(len(stab_inputs)):

        stab_rate_results = stability_rate_deterministic(entailment_model, **stab_inputs[ii], 
                                           p=p, epsilon=epsilon, 
                                           delta=delta, return_all=True, samples=samples,
                                          counts=counts, exact=exact, entailment_mode=entailment_mode, N=N)
        def process_samples_and_counts(stab_rate_results):
            """
            Process samples and counts from entailment rate results.
            Creates two samples for each perturbation: one with 1 and one with 0,
            and adjusts counts based on probabilities.
            
            Args:
                stab_rate_results: Dictionary containing stability rate results
                
            Returns:
                samples: Processed samples tensor
                counts: Processed counts tensor
                perc_counts: Percentage counts tensor
            """
            # Create two samples for each perturbation: one with 1 and one with 0
            original_samples = torch.tensor(stab_rate_results['samples'])
            y_pertbs = torch.tensor(stab_rate_results['y_pertbs'])
            
            # Duplicate samples to create two versions: one with 1 and one with 0
            samples_with_1 = torch.cat([original_samples, torch.ones_like(y_pertbs)[:,None]], dim=-1)
            samples_with_0 = torch.cat([original_samples, torch.zeros_like(y_pertbs)[:,None]], dim=-1)
            samples = torch.cat([samples_with_1, samples_with_0], dim=0)
            
            # Adjust counts based on probabilities
            original_counts = torch.tensor(stab_rate_results['counts'])
            
            # First, scale both y_pertbs and (1-y_pertbs) to ensure minimum value is at least 1
            # while preserving their relative proportions
            y_probs = torch.stack([y_pertbs, 1 - y_pertbs], dim=1)  # Shape: [batch_size, 2]
            
            # Find minimum non-zero value across both probabilities
            min_nonzero = float('inf')
            for i in range(len(y_probs)):
                for j in range(2):
                    if y_probs[i, j] > 0 and y_probs[i, j] < min_nonzero:
                        min_nonzero = y_probs[i, j].item()
            
            if min_nonzero == float('inf'):
                min_nonzero = 1.0
            
            # Scale up probabilities while preserving ratios
            scale_factor = 1.0 / min_nonzero if min_nonzero < 1 else 1.0
            y_pertbs_scaled = y_pertbs * scale_factor
            one_minus_y_pertbs_scaled = (1 - y_pertbs) * scale_factor
            
            # Now use the scaled probabilities to calculate counts
            counts_with_1 = (original_counts * y_pertbs_scaled).ceil().long()
            counts_with_0 = (original_counts * one_minus_y_pertbs_scaled).ceil().long()
            
            # Ensure no counts are zero if original count wasn't zero and probability is non-zero
            counts_with_1 = torch.maximum(counts_with_1, (original_counts > 0).long() * (y_pertbs > 0).long())
            counts_with_0 = torch.maximum(counts_with_0, (original_counts > 0).long() * ((1 - y_pertbs) > 0).long())
            
            counts = torch.cat([counts_with_1, counts_with_0], dim=0)
            
            # Update percentage counts
            perc_counts = counts / counts.sum()
            
            return samples, counts, perc_counts
            
        # Process samples and counts
        samples, counts, perc_counts = process_samples_and_counts(stab_rate_results)

        # stab_rate_results_all_hyps[ii] = stab_rate_results

        stab_rate_results_all_hyps.append({
            'inputs': stab_inputs[ii],
            'stab_rate_results': stab_rate_results,
        })
    return stab_rate_results_all_hyps