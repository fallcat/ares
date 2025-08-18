import itertools
import torch
import math

def sample_s_pertbs(
    s: torch.LongTensor,
    q: float,
    num_samples: int,
    exact: bool = False
):
    original_shape = s.shape
    s = s.view(-1)
    n = s.shape[-1]

    # existing non-exact sampling
    samples = (torch.rand(num_samples, n).to(s.device) > q) * s
    unique_samples, counts = samples.unique(dim=0, return_counts=True)

    if len(unique_samples) == 2**n:
        exact = True

    if exact:
        # Enumerate all 2^|s| subsets explicitly
        s_np = s.cpu().numpy()
        all_combinations = list(itertools.product([0, 1], repeat=n))
        samples = torch.tensor(all_combinations, device=s.device, dtype=torch.long)

        # Compute log probability of each sample explicitly to ensure numerical stability
        log_prob = []
        for comb in samples:
            comb_log_prob = 0.0
            for orig_bit, pertb_bit in zip(s_np, comb.cpu().numpy()):
                if orig_bit == 1:
                    comb_log_prob += torch.log(torch.tensor(1 - q if pertb_bit == 1 else q))
                else:
                    if pertb_bit != 0:
                        comb_log_prob = float('-inf')
                        break
            log_prob.append(comb_log_prob)

        log_prob = torch.tensor(log_prob, device=s.device, dtype=torch.float)
        valid_mask = log_prob != float('-inf')

        samples = samples[valid_mask]
        log_prob = log_prob[valid_mask]
        prob = torch.exp(log_prob - torch.logsumexp(log_prob, dim=0))  # Correct stable normalization

        # Scale counts to ensure minimum count is at least 1
        min_nonzero_prob = prob[prob > 0].min()
        counts = (prob / min_nonzero_prob).ceil().long()

        return samples, counts, prob, exact

    else:
        
        return unique_samples, counts, counts / counts.sum(), exact

def entailment_rate(
    f,
    premises: list, # this is NOT batched
    hypothesis: str, # this is NOT batched
    q: float, # percent of times corrupted 
    epsilon: float = 0.1, 
    delta: float = 0.1, 
    batch_size: int=16, 
    return_all: bool=False,
    verbose=False,
    samples=None, # allow passing in existing samples, this would do for intermediate claims, whose entailment is dependent on previous claims
    counts=None, # if samples is not None, then if counts is not None, use this counts, or else treat all samples the same (counts be 1 for all).
    exact: bool = False
):
    n = len(premises)
    s = torch.ones(n)
    
    # reference prediction
    y = f([premises], [hypothesis], [s]) # can we cache this?
    
    N = int(math.log(2/delta) / (2 * (epsilon**2))) + 1
    if verbose:
        print('N', N)
    
    all_y_pertbs = []
    
    if samples is None:
        samples, counts, perc_counts, exact = sample_s_pertbs(s, q, N, exact=exact)
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
            s=s_pertbs
        )
        all_y_pertbs.append(y_pertbs)
        
    all_y_pertbs = torch.cat(all_y_pertbs, dim=0)

    ent_rate = ((y == all_y_pertbs).float() * perc_counts).sum()
    # print('ent_rate', ent_rate)
    # import pdb; pdb.set_trace()
    
    if return_all:
        return {
            'entailment_rate': ent_rate.item(),
            'y': y.tolist(),
            'y_pertbs': all_y_pertbs.tolist(),
            'samples': samples.tolist(),
            'counts': counts.tolist(),
            'perc_counts': perc_counts.tolist(),
            'exact': exact # bool: True: doing exact all combinations, False: doing sampling
        }
    else:
        return ent_rate

def tree_entailment_rate(
    entailment_model, 
    ent_inputs,
    children=None,
    parents=None,
    exact=True, 
    q = 0.5,
    epsilon=0.1,
    delta=0.1
):
    ent_rate_results_all_hyps = []

    samples = None
    counts = None
    perc_counts = None
    for ii in range(len(ent_inputs)):

        ent_rate_results = entailment_rate(entailment_model, **ent_inputs[ii], 
                                           q=q, epsilon=epsilon, 
                                           delta=delta, return_all=True, samples=samples,
                                          counts=counts, exact=exact)
        samples = torch.cat([torch.tensor(ent_rate_results['samples']), 
                             torch.tensor(ent_rate_results['y_pertbs'])[:,None]], dim=-1)
        counts = torch.tensor(ent_rate_results['counts'])
        perc_counts = torch.tensor(ent_rate_results['perc_counts'])

        # ent_rate_results_all_r[r] = ent_rate_results

        ent_rate_results_all_hyps.append({
            'inputs': ent_inputs[ii],
            'ent_rate_results': ent_rate_results,
        })
    return ent_rate_results_all_hyps