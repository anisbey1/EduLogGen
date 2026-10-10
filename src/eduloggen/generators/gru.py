"""Recurrent neural generator: a GRU over event types with a timing head.

Optional: requires PyTorch (``pip install "eduloggen[neural]"``).

Fit:
    A single-layer GRU reads a session's event types and, at each step,
    predicts the next event type (softmax) and the time spent after the
    current event before the next one. The time spent therefore depends on
    the whole history of the session, not only on the current event type as
    in ``semi_markov``. Timing head ``binned`` (default): a softmax over
    quantile bins of the training gaps (one bin per distinct value when there
    are few, e.g. whole days), and a gap is drawn from the training values in
    the chosen bin, so discrete gap values are reproduced exactly;
    ``mixture``: a mixture of log-normal distributions (mixture density
    network). Training minimises the summed negative log-likelihood
    with Adam, on sessions truncated to ``max_length`` events, with early
    stopping on a seeded validation split of sessions. Training is seeded,
    deterministic, and single-threaded by default.

Generate:
    Session lengths, learners, and start times come from the shared machinery
    (as for every generator). Tokens and gaps are drawn step by step from the
    network's outputs with the generator's own random number generator, so a
    model and a seed always give the same output.

Model files stay JSON: weights are stored as base64-encoded float32 arrays.

Hyperparameters:
    ``embedding_dim`` (32), ``hidden_dim`` (64), ``timing`` (``binned`` or
    ``mixture``), ``n_bins`` (32), ``n_mixtures`` (3),
    ``epochs`` (30), ``batch_size`` (64), ``learning_rate`` (0.003),
    ``max_length`` (256), ``validation_fraction`` (0.1), ``patience`` (3),
    ``seed`` (0), ``threads`` (1), plus ``length_model`` and ``fixed_length``.
"""

from __future__ import annotations

import base64
import logging
import math
import random
from array import array
from bisect import bisect_left
from collections.abc import Mapping
from itertools import pairwise
from typing import Any, ClassVar

from eduloggen.core import FitError, GenerationError
from eduloggen.generators.base import (
    BaseGenerator,
    SequenceSampler,
    _bad,
    _choice,
    _number,
)
from eduloggen.generators.distributions import sketch
from eduloggen.models import Dataset, GeneratorModel

__all__ = ["GRUGenerator"]

logger = logging.getLogger(__name__)

_LOG_MIN_GAP = math.log(1e-3)


def _torch(error: type[FitError] | type[GenerationError]) -> Any:
    try:
        import torch
    except ImportError:
        raise error(
            "the gru generator requires PyTorch; install it with: "
            'pip install "eduloggen[neural]"',
            code="missing_dependency",
            context={"package": "torch", "extra": "neural"},
        ) from None
    return torch


class GRUGenerator(BaseGenerator):
    """Sessions from a GRU language model over event types with timing."""

    name: ClassVar[str] = "gru"
    version: ClassVar[str] = "1.0"
    tags: ClassVar[frozenset[str]] = frozenset(
        {"probabilistic", "sequence", "supports_timing", "neural"}
    )
    defaults: ClassVar[Mapping[str, Any]] = {
        "embedding_dim": 32,
        "hidden_dim": 64,
        "timing": "binned",
        "n_bins": 32,
        "n_mixtures": 3,
        "epochs": 30,
        "batch_size": 64,
        "learning_rate": 0.003,
        "max_length": 256,
        "validation_fraction": 0.1,
        "patience": 3,
        "seed": 0,
        "threads": 1,
    }

    def _validate_family(self, hyperparameters: dict[str, Any]) -> None:
        for key in (
            "embedding_dim",
            "hidden_dim",
            "n_bins",
            "n_mixtures",
            "epochs",
            "batch_size",
            "max_length",
            "patience",
            "threads",
        ):
            _number(hyperparameters, key, minimum=1, integer=True)
        _choice(hyperparameters, "timing", ("binned", "mixture"))
        _number(hyperparameters, "seed", minimum=0, integer=True)
        _number(hyperparameters, "learning_rate", minimum=1e-9)
        _number(hyperparameters, "validation_fraction", minimum=0.0)
        if hyperparameters["validation_fraction"] >= 1:
            raise _bad("validation_fraction", "must be below 1")

    # -- fit ------------------------------------------------------------------

    def _fit_family(
        self,
        dataset: Dataset,
        sequences: tuple[tuple[str, ...], ...],
        hyperparameters: Mapping[str, Any],
    ) -> dict[str, Any]:
        torch = _torch(FitError)
        hp = hyperparameters
        vocabulary = sorted({t for s in sequences for t in s})
        index = {t: i + 1 for i, t in enumerate(vocabulary)}  # 0 = start
        gaps = _session_gaps(dataset)
        examples = [
            (
                [index[t] for t in seq[: hp["max_length"]]],
                [math.log(max(g, 1e-3)) for g in gap[: hp["max_length"] - 1]],
            )
            for seq, gap in zip(sequences, gaps, strict=True)
        ]
        all_gaps = [g for _, gap in examples for g in gap]
        raw_gaps = [max(g, 1e-3) for gap in gaps for g in gap[: hp["max_length"] - 1]]
        bins = _bins(raw_gaps, hp["n_bins"]) if hp["timing"] == "binned" else None
        timing_size = len(bins["values"]) if bins else 3 * hp["n_mixtures"]
        rng = random.Random(hp["seed"])
        order = list(range(len(examples)))
        rng.shuffle(order)
        n_val = int(len(order) * hp["validation_fraction"])
        val = [examples[i] for i in order[:n_val]]
        train = [examples[i] for i in order[n_val:]] or val

        threads = torch.get_num_threads()
        deterministic = torch.are_deterministic_algorithms_enabled()
        torch.set_num_threads(hp["threads"])
        torch.use_deterministic_algorithms(True)
        try:
            torch.manual_seed(hp["seed"])
            net = _network(torch, len(vocabulary) + 1, timing_size, hp)
            edges = torch.tensor(bins["edges"]) if bins else None
            optimiser = torch.optim.Adam(net.parameters(), lr=hp["learning_rate"])
            best, best_state, stale, history = math.inf, None, 0, []
            for epoch in range(hp["epochs"]):
                net.train()
                rng.shuffle(train)
                for batch in _batches(train, hp["batch_size"]):
                    optimiser.zero_grad()
                    loss = _loss(torch, net, batch, hp, edges)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                    optimiser.step()
                net.eval()
                with torch.no_grad():
                    score = _mean_loss(torch, net, val or train, hp, edges)
                history.append(score)
                if score < best - 1e-6:
                    best, stale = score, 0
                    best_state = {
                        k: v.detach().clone() for k, v in net.state_dict().items()
                    }
                else:
                    stale += 1
                    if stale >= hp["patience"]:
                        break
                logger.info("gru epoch %d: validation loss %.4f", epoch + 1, score)
            assert best_state is not None
        finally:
            torch.set_num_threads(threads)
            torch.use_deterministic_algorithms(deterministic)

        return {
            "gru": {
                "timing_bins": bins,
                "timing_size": timing_size,
                "vocabulary": vocabulary,
                "weights": {k: _encode(v) for k, v in sorted(best_state.items())},
                "epochs_trained": len(history),
                "validation_loss": best,
                "history": history,
                "n_parameters": sum(v.numel() for v in best_state.values()),
                "log_gap_range": (
                    [min(all_gaps), max(all_gaps)] if all_gaps else [0.0, 0.0]
                ),
            }
        }

    def _describe_family(self, model: GeneratorModel) -> dict[str, Any]:
        gru = model.parameters["gru"]
        return {
            "n_parameters": gru["n_parameters"],
            "epochs_trained": gru["epochs_trained"],
            "validation_loss": gru["validation_loss"],
        }

    # -- generate -------------------------------------------------------------

    def _sequence_sampler(self, model: GeneratorModel) -> SequenceSampler:
        torch = _torch(GenerationError)
        try:
            gru = model.parameters["gru"]
            vocabulary = list(gru["vocabulary"])
            bins = gru["timing_bins"]
            net = _network(
                torch, len(vocabulary) + 1, gru["timing_size"], model.hyperparameters
            )
            net.load_state_dict(
                {k: _decode(torch, v) for k, v in gru["weights"].items()}
            )
            low, high = gru["log_gap_range"]
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise GenerationError(
                "malformed gru model", code="generation_invalid_model"
            ) from exc
        net.eval()
        k = model.hyperparameters["n_mixtures"]

        def sample(rng: random.Random, length: int) -> tuple[list[str], list[float]]:
            tokens: list[str] = []
            gaps: list[float] = []
            state = None
            current = 0
            with torch.no_grad():
                for step in range(length):
                    x = net.embed(torch.tensor([[current]]))
                    out, state = net.gru(x, state)
                    h = out[0, -1]
                    if step and bins:
                        weights = torch.softmax(net.timing(h), 0).tolist()
                        values = bins["values"][_draw(rng, weights)]
                        gaps.append(values[rng.randrange(len(values))])
                    elif step:
                        mix = net.timing(h)
                        weights = torch.softmax(mix[:k], 0).tolist()
                        means = mix[k : 2 * k].tolist()
                        sigmas = torch.exp(mix[2 * k :].clamp(-5.0, 3.0)).tolist()
                        c = _draw(rng, weights)
                        log_gap = min(max(rng.gauss(means[c], sigmas[c]), low), high)
                        gaps.append(math.exp(log_gap))
                    probs = torch.softmax(net.next_token(h)[1:], 0).tolist()
                    current = _draw(rng, probs) + 1
                    tokens.append(vocabulary[current - 1])
            return tokens, gaps

        return sample


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _network(torch: Any, n_tokens: int, timing_size: int, hp: Mapping[str, Any]) -> Any:
    nn = torch.nn

    class Net(nn.Module):  # type: ignore[misc,name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(n_tokens, hp["embedding_dim"])
            self.gru = nn.GRU(hp["embedding_dim"], hp["hidden_dim"], batch_first=True)
            self.next_token = nn.Linear(hp["hidden_dim"], n_tokens)
            self.timing = nn.Linear(hp["hidden_dim"], timing_size)

    return Net()


def _session_gaps(dataset: Dataset) -> list[list[float]]:
    by_session: dict[str, list[Any]] = {}
    for event in dataset.events:
        if event.session_id is not None:
            by_session.setdefault(event.session_id, []).append(event)
    out = []
    # same order as session_sequences(): by session id
    for session in sorted(dataset.sessions or (), key=lambda s: s.session_id):
        events = sorted(
            by_session.get(session.session_id, ()), key=lambda e: e.timestamp
        )
        out.append(
            [(b.timestamp - a.timestamp).total_seconds() for a, b in pairwise(events)]
        )
    return out


def _batches(
    examples: list[tuple[list[int], list[float]]], size: int
) -> list[list[tuple[list[int], list[float]]]]:
    return [examples[i : i + size] for i in range(0, len(examples), size)]


def _loss(
    torch: Any,
    net: Any,
    batch: list[tuple[list[int], list[float]]],
    hp: Mapping[str, Any],
    edges: Any = None,
) -> Any:
    k = hp["n_mixtures"]
    longest = max(len(tokens) for tokens, _ in batch)
    inputs = torch.zeros((len(batch), longest), dtype=torch.long)
    targets = torch.zeros((len(batch), longest), dtype=torch.long)
    gap_targets = torch.zeros((len(batch), longest))
    token_mask = torch.zeros((len(batch), longest))
    gap_mask = torch.zeros((len(batch), longest))
    for row, (tokens, gaps) in enumerate(batch):
        n = len(tokens)
        inputs[row, 1:n] = torch.tensor(tokens[:-1], dtype=torch.long)
        targets[row, :n] = torch.tensor(tokens, dtype=torch.long)
        token_mask[row, :n] = 1.0
        if gaps:
            # step j (input = token j) predicts the gap after token j
            gap_targets[row, 1 : 1 + len(gaps)] = torch.tensor(gaps)
            gap_mask[row, 1 : 1 + len(gaps)] = 1.0
    out, _ = net.gru(net.embed(inputs))
    logits = net.next_token(out)
    token_nll = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
    ).reshape(targets.shape)
    mix = net.timing(out)
    if edges is not None:
        bin_targets = torch.bucketize(gap_targets, edges)
        gap_nll = torch.nn.functional.cross_entropy(
            mix.reshape(-1, mix.shape[-1]), bin_targets.reshape(-1), reduction="none"
        ).reshape(bin_targets.shape)
    else:
        log_w = torch.log_softmax(mix[..., :k], -1)
        mean = mix[..., k : 2 * k]
        log_s = mix[..., 2 * k :].clamp(-5.0, 3.0)
        z = (gap_targets.unsqueeze(-1) - mean) / torch.exp(log_s)
        log_p = log_w - log_s - 0.5 * z**2 - 0.5 * math.log(2 * math.pi)
        gap_nll = -torch.logsumexp(log_p, -1)
    total = (token_nll * token_mask).sum() + (gap_nll * gap_mask).sum()
    return total / token_mask.sum()


def _mean_loss(
    torch: Any,
    net: Any,
    examples: list[tuple[list[int], list[float]]],
    hp: Mapping[str, Any],
    edges: Any = None,
) -> float:
    total, weight = 0.0, 0
    for batch in _batches(examples, hp["batch_size"]):
        n = sum(len(tokens) for tokens, _ in batch)
        total += float(_loss(torch, net, batch, hp, edges)) * n
        weight += n
    return total / max(weight, 1)


def _bins(gaps: list[float], n_bins: int) -> dict[str, Any]:
    """Bins of log gaps: one per distinct value if few, else quantile bins.

    ``edges`` are the inner bin boundaries in log seconds (a gap falls in
    bin ``i`` when ``edges[i-1] < log(gap) <= edges[i]``); ``values`` holds,
    per bin, a sketch of the training gaps in seconds that fell in it.
    """
    if not gaps:
        return {"edges": [], "values": [[1.0]]}
    logs = sorted(math.log(g) for g in gaps)
    distinct = sorted(set(logs))
    if len(distinct) <= n_bins:
        edges = [(a + b) / 2 for a, b in pairwise(distinct)]
    else:
        quantiles = [
            logs[round(i * (len(logs) - 1) / n_bins)] for i in range(1, n_bins)
        ]
        edges = sorted(set(quantiles))
    members: list[list[float]] = [[] for _ in range(len(edges) + 1)]
    for gap in gaps:
        members[bisect_left(edges, math.log(gap))].append(gap)
    values = [
        sketch(m, 64) or [math.exp(e)]
        for m, e in zip(members, [*edges, edges[-1] if edges else 0.0], strict=True)
    ]
    return {"edges": edges, "values": values}


def _encode(tensor: Any) -> dict[str, Any]:
    data = array("f", tensor.detach().reshape(-1).tolist()).tobytes()
    return {
        "shape": list(tensor.shape),
        "float32": base64.b64encode(data).decode("ascii"),
    }


def _decode(torch: Any, item: Mapping[str, Any]) -> Any:
    values = array("f")
    values.frombytes(base64.b64decode(item["float32"]))
    return torch.tensor(values.tolist(), dtype=torch.float32).reshape(item["shape"])


def _draw(rng: random.Random, weights: list[float]) -> int:
    threshold = rng.random() * sum(weights)
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if threshold < acc:
            return i
    return len(weights) - 1
