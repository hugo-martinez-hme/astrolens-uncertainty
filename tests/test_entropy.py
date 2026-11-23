import math

import torch

from astrolens.uncertainty.entropy import summarize_entropy


def test_entropy_distinguishes_disagreement_at_equal_mean():
    # Each pass contains predictions for the same two images, A and B.
    samples = torch.tensor(
        [
            [[0.5, 0.5, 0.0], [1.0, 0.0, 0.0]],  # Pass 1
            [[0.5, 0.5, 0.0], [0.0, 1.0, 0.0]],  # Pass 2
        ]
    )

    summary = summarize_entropy(samples)
    log_two = math.log(2)

    torch.testing.assert_close(summary.predictive_entropy, torch.tensor([log_two, log_two]))
    torch.testing.assert_close(summary.expected_entropy, torch.tensor([log_two, 0.0]))
    torch.testing.assert_close(summary.mutual_information, torch.tensor([0.0, log_two]))
