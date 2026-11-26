import random


def cohen_kappa(a, b):
    """Agreement between two raters beyond what chance would give.

    a, b: two equally long lists of labels for the same items.
      p_o = observed share of items on which the raters agree
      p_e = sum over the labels c of (share of c in a) * (share of c in b)       agreement expected by chance
      kappa = (p_o - p_e) / (1 - p_e)          (return 1.0 if p_e == 1)
    """
    # TODO 1
    raise NotImplementedError


def bootstrap_ci(values, n_resamples=2000, level=0.95, seed=0):
    """A confidence interval for the mean of `values` by the percentile bootstrap.

    Use rng = random.Random(seed). Repeat n_resamples times: draw len(values) items WITH replacement
    (rng.choice) and take their mean. Sort the means and return
        (means[int((1 - level) / 2 * n_resamples)],  means[int((1 + level) / 2 * n_resamples) - 1])
    """
    # TODO 2
    raise NotImplementedError


def paired_bootstrap(scores_a, scores_b, n_resamples=2000, seed=0):
    """Two systems scored on the SAME items. In what share of resamples is the total of A above the total of B?

    Use rng = random.Random(seed). Each resample draws n item indices with replacement (rng.randrange(n))
    and uses the same indices for both systems, which keeps the pairing.
    A value near 1 means A is reliably better on this kind of item; a value near 0.5 means the data cannot tell.
    """
    # TODO 3
    raise NotImplementedError
