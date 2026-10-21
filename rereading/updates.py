"""Ways of turning a batch into a weight change, for the rereading runs.

    plain                 the average loss over the batch's sequences
    neighbours            each sequence's loss weighted by how many gradient neighbours it has:
                          other sequences in the batch whose last-layer gradient points the same
                          way. A sequence that needs a change no other sequence needs is a
                          one-off, and gets weight 0; the more others share its change, the more
                          weight it gets.
    shuffled neighbours   the same weights, handed to the batch's sequences in a random order.
                          A control: if it does as well as neighbours, what helped was the
                          reweighting itself, not who has neighbours.
    batch agreement       each sequence's loss weighted by how well its gradient agrees with the
                          batch's average gradient: agreement with everyone, not with a group.
                          A control: if it does as well as neighbours, finding the groups did
                          not matter.

The last layer's gradient for one token is (its error) x (its features): the error is the guessed
probabilities minus 1 at the true next token, the features are what the network feeds the last
layer. A sequence's last-layer gradient is the sum of these over its tokens, VOCABULARY x width
numbers. The inner product of two sequences' gradients is summed one slice of the vocabulary at
a time, so the full gradients never sit in memory at once. The slices are multiplied in 16-bit
bfloat16 (a GPU does that several times faster) and added up in 32-bit; the cosines then match
the exact ones to about 0.01 (test_rereading.py, and look_at_neighbours.py on the GPU).

Two sequences are neighbours when their gradients point together unusually closely for this
batch: their cosine is more than `spreads` standard deviations above the average cosine of the
batch's pairs. A fixed cosine threshold would not do: as training goes on, every pair's cosine
shrinks toward 0, and the method would then stop learning altogether rather than stop learning
from one-offs.
"""

import torch
import torch.nn.functional as F

VOCABULARY_SLICE = 8192


def sequence_losses(logits, targets):
    """[sequence]: the average next-token loss of each sequence in the batch."""
    losses = F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten(), reduction="none")
    return losses.view(targets.shape).mean(dim=1)


def log_totals(logits):
    """[sequence, token, 1]: log of the sum of exp(logits) over the vocabulary, one slice at a time."""
    top = logits.max(dim=-1, keepdim=True).values.float()
    total = torch.zeros_like(top)
    for start in range(0, logits.shape[-1], VOCABULARY_SLICE):
        total += (logits[..., start:start + VOCABULARY_SLICE].float() - top).exp().sum(dim=-1, keepdim=True)
    return top + total.log()


def gradient_inner_products(features, logits, targets, exact=False):
    """[sequence, sequence]: the inner product of each pair of sequences' last-layer gradients.

    exact: multiply in 32-bit instead of bfloat16, to check how much bfloat16 changes.
    """
    kind = torch.float32 if exact else torch.bfloat16
    with torch.no_grad():
        totals = log_totals(logits)
        inner = torch.zeros(len(logits), len(logits), device=logits.device)
        for start in range(0, logits.shape[-1], VOCABULARY_SLICE):
            errors = (logits[..., start:start + VOCABULARY_SLICE].float() - totals).exp()
            hit = (targets >= start) & (targets < start + errors.shape[-1])
            errors.scatter_add_(-1, (targets - start).clamp(0, errors.shape[-1] - 1).unsqueeze(-1),
                                -hit.unsqueeze(-1).float())
            gradient = torch.einsum("btv,btd->bvd", errors.to(kind), features.to(kind)).flatten(1).float()
            inner += gradient @ gradient.T
        return inner


def cosines(inner, centred):
    """[sequence, sequence]: cosines from inner products.

    centred: first subtract the batch's average gradient, so that the change the whole batch
    agrees on does not make every pair look alike.
    """
    if centred:  # the inner products of the gradients minus their batch average
        row = inner.mean(dim=0, keepdim=True)
        inner = inner - row - row.T + inner.mean()
    length = inner.diagonal().clamp(min=1e-20).sqrt()
    return inner / length[:, None] / length[None, :]


def neighbour_pairs(cosine, spreads):
    """[sequence, sequence]: True where two different sequences of the batch are neighbours."""
    others = ~torch.eye(len(cosine), dtype=torch.bool, device=cosine.device)
    pairs = cosine[others]
    return ((cosine - pairs.mean()) / pairs.std() > spreads) & others


def weights_from(scores):
    """Scores scaled so the batch's weights average 1, or all 1 if every score is 0. Also says which happened."""
    if scores.sum() > 0:
        return scores / scores.mean(), False
    return torch.ones_like(scores), True


def neighbour_weights(cosine, spreads):
    """[sequence], and whether the batch fell back to plain weights: neighbour counts, averaging 1."""
    return weights_from(neighbour_pairs(cosine, spreads).sum(dim=1).float())


def batch_agreement_weights(inner):
    """[sequence], and whether the batch fell back: each gradient's cosine with the batch's average
    gradient, negative ones counted as 0, averaging 1."""
    average = inner.mean(dim=1)                   # inner product with the average gradient
    length = inner.diagonal().clamp(min=1e-20).sqrt()
    average_length = inner.mean().clamp(min=1e-20).sqrt()
    return weights_from((average / length / average_length).clamp(min=0))
