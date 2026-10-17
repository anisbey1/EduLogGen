"""Time-aware autoregressive Transformer generator (optional; PyTorch).

Requires PyTorch: ``pip install "eduloggen[neural]"``.

Model:
    A decoder-only causal Transformer over a session. For a session with
    events ``e1..eT`` and gaps ``g1..g(T-1)`` (``g_t`` is the time from
    ``e_t`` to ``e_(t+1)``, the convention of every EduLogGen generator),
    position ``p`` receives

    - the event embedding of ``BOS`` (``p = 0``) or of ``e_p``;
    - the embedding of the gap bin of ``g_(p-1)``, the gap that led to
      ``e_p``; positions 0 and 1 receive a reserved "no gap" id, so ``BOS``
      never implies an invented gap;
    - a learned position embedding (relative to the context window).

    From the hidden state at ``p`` the model predicts the next event
    ``e_(p+1)`` (softmax over the vocabulary) and, for ``p >= 1``, the gap
    ``g_p`` after the current event (softmax over gap bins), exactly as the
    ``gru`` generator does: ``P(e_(p+1), g_p | history) = P(e_(p+1) | h_p)
    P(g_p | h_p)``. The time head does not see the sampled next event. Gap
    bins are fitted on training gaps (one bin per distinct value when there
    are few, such as whole days, otherwise quantile bins); a sampled bin is
    turned into a gap by drawing one of the training gaps stored for that
    bin, so discrete gaps are reproduced exactly. Zero gaps count as 1 ms.

    Blocks are pre-norm: causal multi-head self-attention, residual
    connections, layer normalisation, a position-wise feed-forward network,
    and dropout (training only). Padding is masked out of attention and of
    both losses. Loss: ``L_event + time_loss_weight * L_time`` (mean
    cross-entropies over real targets).

Training:
    Inner training and early-stopping validation groups are split by
    **learner**. Event vocabulary comes from the data being fitted; gap bins
    from the inner training learners only. Sessions longer than
    ``max_seq_len`` are split into overlapping windows (``window_overlap``),
    each target being scored exactly once, in the first window that contains
    it after the overlap, so no training target is discarded; positions are
    relative to the window. AdamW, gradient clipping, early stopping on the
    validation loss, best-epoch checkpoint; a non-finite loss stops training
    with an error. Training is seeded and, on CPU, deterministic under the
    tested PyTorch configuration.

Generate:
    Session lengths, learners, and start times come from the shared
    machinery, as for every generator (no end-of-session token). Events and
    gaps are sampled step by step with the generator's own random number
    generator (``temperature``, optional ``top_k``) on one CPU thread, with a
    key/value cache; beyond ``max_seq_len`` events the context rolls over the last
    ``max_seq_len`` positions while the whole session is kept in the output.

The model sees one session at a time: it captures long within-session
dependencies, not dependencies between sessions of the same learner.

Model files stay JSON (weights as base64 float32 arrays).
"""

from __future__ import annotations

import logging
import math
import random
from bisect import bisect_left
from collections.abc import Mapping
from typing import Any, ClassVar

from eduloggen.core import FitError, GenerationError
from eduloggen.generators._neural import (
    _bins,
    _decode,
    _deterministic,
    _device,
    _draw,
    _encode,
    _learner_validation,
    _session_gaps,
    _session_learners,
    _torch,
)
from eduloggen.generators.base import (
    BaseGenerator,
    SequenceSampler,
    _bad,
    _choice,
    _number,
)
from eduloggen.models import Dataset, GeneratorModel

__all__ = ["TransformerGenerator"]

logger = logging.getLogger(__name__)

PAD = 0
BOS = 1
NO_GAP = 1
_FIRST_EVENT = 2  # event ids start here
_FIRST_BIN = 2  # gap-input ids start here (0 = pad, 1 = no gap)

Example = tuple[list[int], list[int], list[int], list[int]]
"""Window: event inputs, gap inputs, event targets, gap targets (-1 = none)."""


class TransformerGenerator(BaseGenerator):
    """Sessions from a time-aware causal Transformer over event types."""

    name: ClassVar[str] = "transformer"
    version: ClassVar[str] = "1.0"
    tags: ClassVar[frozenset[str]] = frozenset(
        {"probabilistic", "sequence", "supports_timing", "neural"}
    )
    defaults: ClassVar[Mapping[str, Any]] = {
        "d_model": 128,
        "num_heads": 4,
        "num_layers": 2,
        "ff_dim": 256,
        "dropout": 0.1,
        "learning_rate": 0.001,
        "weight_decay": 0.01,
        "batch_size": 64,
        "epochs": 100,
        "patience": 10,
        "max_seq_len": 256,
        "window_overlap": 0.5,
        "time_loss_weight": 1.0,
        "n_bins": 32,
        "validation_fraction": 0.1,
        "grad_clip": 1.0,
        "temperature": 1.0,
        "top_k": 0,
        "device": "cpu",
        "seed": 0,
        "threads": 1,
    }

    def _validate_family(self, hp: dict[str, Any]) -> None:
        for key in (
            "d_model",
            "num_heads",
            "num_layers",
            "ff_dim",
            "batch_size",
            "epochs",
            "patience",
            "max_seq_len",
            "n_bins",
            "threads",
        ):
            _number(hp, key, minimum=1, integer=True)
        for key in ("seed", "top_k"):
            _number(hp, key, minimum=0, integer=True)
        for key in ("learning_rate", "grad_clip", "temperature"):
            _number(hp, key, minimum=1e-9)
        for key in ("weight_decay", "time_loss_weight", "validation_fraction"):
            _number(hp, key, minimum=0.0)
        _number(hp, "dropout", minimum=0.0)
        _number(hp, "window_overlap", minimum=0.0)
        if hp["dropout"] >= 1:
            raise _bad("dropout", "must be below 1")
        if hp["window_overlap"] >= 1:
            raise _bad("window_overlap", "must be below 1")
        if hp["validation_fraction"] >= 1:
            raise _bad("validation_fraction", "must be below 1")
        if hp["d_model"] % hp["num_heads"]:
            raise _bad("d_model", "must be divisible by num_heads")
        if hp["max_seq_len"] < 2:
            raise _bad("max_seq_len", "must be at least 2")
        _choice(hp, "device", ("cpu", "cuda", "mps", "auto"))

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
        index = {t: i + _FIRST_EVENT for i, t in enumerate(vocabulary)}
        gaps = [[max(g, 1e-3) for g in gap] for gap in _session_gaps(dataset)]
        rng = random.Random(hp["seed"])
        learners = _session_learners(dataset)
        held = _learner_validation(learners, hp["validation_fraction"], rng)
        train_ids = [i for i in range(len(sequences)) if i not in held]
        val_ids = sorted(held)
        bins = _bins([g for i in train_ids for g in gaps[i]], hp["n_bins"])
        n_bins = len(bins["values"])

        def windows(ids: list[int]) -> list[Example]:
            out: list[Example] = []
            for i in ids:
                events = [index[t] for t in sequences[i]]
                out += _windows(events, _bin_ids(gaps[i], bins["edges"]), hp)
            return out

        train, val = windows(train_ids), windows(val_ids)
        if not train:
            raise FitError("no training sessions", code="fit_empty_corpus")
        if not val:
            logger.warning(
                "transformer: no validation learners; training for %d epochs "
                "without early stopping",
                hp["epochs"],
            )
        device = _device(torch, hp["device"])

        threads = torch.get_num_threads()
        deterministic = torch.are_deterministic_algorithms_enabled()
        torch.set_num_threads(hp["threads"])
        _deterministic(torch, device)
        try:
            torch.manual_seed(hp["seed"])
            net = _network(torch, len(vocabulary) + _FIRST_EVENT, n_bins, hp).to(device)
            optimiser = torch.optim.AdamW(
                net.parameters(),
                lr=hp["learning_rate"],
                weight_decay=hp["weight_decay"],
            )
            history: list[dict[str, float]] = []
            best, best_epoch, best_state, stale = math.inf, 0, None, 0
            for epoch in range(1, hp["epochs"] + 1):
                net.train()
                rng.shuffle(train)
                running, seen = 0.0, 0
                for batch in _batches(train, hp["batch_size"]):
                    optimiser.zero_grad()
                    loss = _loss(torch, net, batch, hp, device)
                    if not torch.isfinite(loss):
                        raise FitError(
                            f"transformer training diverged at epoch {epoch} "
                            "(non-finite loss); lower learning_rate",
                            code="fit_diverged",
                            context={"epoch": epoch},
                        )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), hp["grad_clip"])
                    optimiser.step()
                    running += float(loss) * len(batch)
                    seen += len(batch)
                # running mean of the epoch's training batches (dropout on);
                # early stopping uses the validation loss (dropout off)
                train_loss = running / max(seen, 1)
                net.eval()
                with torch.no_grad():
                    val_loss = (
                        _mean_loss(torch, net, val, hp, device) if val else train_loss
                    )
                history.append({"train": train_loss, "validation": val_loss})
                logger.info(
                    "transformer epoch %d: train %.4f, validation %.4f",
                    epoch,
                    train_loss,
                    val_loss,
                )
                if val_loss < best - 1e-6:
                    best, best_epoch, stale = val_loss, epoch, 0
                    best_state = {
                        k: v.detach().to("cpu").clone()
                        for k, v in net.state_dict().items()
                    }
                elif val:
                    stale += 1
                    if stale >= hp["patience"]:
                        break
            assert best_state is not None
        finally:
            torch.set_num_threads(threads)
            torch.use_deterministic_algorithms(deterministic)

        longest = max(len(s) for s in sequences)
        return {
            "transformer": {
                "vocabulary": vocabulary,
                "timing_bins": bins,
                "n_bins": n_bins,
                "weights": {k: _encode(v) for k, v in sorted(best_state.items())},
                "n_parameters": sum(v.numel() for v in best_state.values()),
                "training": {
                    "history": history,
                    "best_epoch": best_epoch,
                    "epochs_trained": len(history),
                    "best_validation_loss": best,
                    "train_learners": len({learners[i] for i in train_ids}),
                    "validation_learners": len({learners[i] for i in val_ids}),
                    "train_sessions": len(train_ids),
                    "validation_sessions": len(val_ids),
                    "train_windows": len(train),
                    "longest_session": longest,
                    "context_length": min(longest, hp["max_seq_len"]),
                    "device": str(device),
                    "torch_version": str(torch.__version__),
                },
            }
        }

    def _describe_family(self, model: GeneratorModel) -> dict[str, Any]:
        params = model.parameters["transformer"]
        training = params["training"]
        return {
            "n_parameters": params["n_parameters"],
            "epochs_trained": training["epochs_trained"],
            "best_epoch": training["best_epoch"],
            "best_validation_loss": training["best_validation_loss"],
            "context_length": training["context_length"],
        }

    # -- generate -------------------------------------------------------------

    def _sequence_sampler(self, model: GeneratorModel) -> SequenceSampler:
        torch = _torch(GenerationError)
        hp = model.hyperparameters
        try:
            params = model.parameters["transformer"]
            vocabulary = list(params["vocabulary"])
            bins = params["timing_bins"]
            net = _network(torch, len(vocabulary) + _FIRST_EVENT, params["n_bins"], hp)
            net.load_state_dict(
                {k: _decode(torch, v) for k, v in params["weights"].items()}
            )
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise GenerationError(
                "malformed transformer model", code="generation_invalid_model"
            ) from exc
        net.eval()
        window = hp["max_seq_len"]
        temperature, top_k = hp["temperature"], hp["top_k"]

        def sample(rng: random.Random, length: int) -> tuple[list[str], list[float]]:
            threads = torch.get_num_threads()
            torch.set_num_threads(1)  # one step is too small to parallelise
            try:
                return _sample(rng, length)
            finally:
                torch.set_num_threads(threads)

        def _sample(rng: random.Random, length: int) -> tuple[list[str], list[float]]:
            tokens: list[str] = []
            gaps: list[float] = []
            events_in, gaps_in = [BOS], [NO_GAP]
            cache: list[Any] | None = None
            with torch.no_grad():
                for step in range(length):
                    if len(events_in) <= window:
                        hidden, cache = _step(torch, net, events_in, gaps_in, cache)
                    else:  # rolling context: recompute the last `window` steps
                        hidden, _ = _step(
                            torch, net, events_in[-window:], gaps_in[-window:], None
                        )
                    if step:  # the gap after the current event
                        weights = _weights(
                            torch, net["time_head"](hidden), temperature, top_k
                        )
                        b = _draw(rng, weights)
                        values = bins["values"][b]
                        gaps.append(values[rng.randrange(len(values))])
                    logits = net["event_head"](hidden)[_FIRST_EVENT:]
                    e = _draw(rng, _weights(torch, logits, temperature, top_k))
                    tokens.append(vocabulary[e])
                    events_in.append(e + _FIRST_EVENT)
                    gaps_in.append(b + _FIRST_BIN if step else NO_GAP)
            return tokens, gaps

        return sample


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def _bin_ids(gaps: list[float], edges: list[float]) -> list[int]:
    """Bin index of each gap (same rule as the gap bins)."""
    return [bisect_left(edges, math.log(g)) for g in gaps]


def _windows(
    events: list[int], bins: list[int], hp: Mapping[str, Any]
) -> list[Example]:
    """Split one session into training windows; every target appears once.

    Full-session arrays (position ``p`` = input ``BOS`` or ``e_p``):

    - event inputs ``[BOS, e1, ..., e(T-1)]`` and targets ``[e1, ..., eT]``;
    - gap inputs ``[no gap, no gap, bin(g1), ..., bin(g(T-2))]``;
    - gap targets ``[-1, bin(g1), ..., bin(g(T-1))]`` (no gap target at
      ``BOS``).
    """
    n = len(events)
    ev_in = [BOS, *events[:-1]]
    gap_in = [NO_GAP, NO_GAP, *[b + _FIRST_BIN for b in bins[:-1]]][:n]
    ev_out = list(events)
    gap_out = [-1, *bins][:n]
    size = hp["max_seq_len"]
    if n <= size:
        return [(ev_in, gap_in, ev_out, gap_out)]
    overlap = int(size * hp["window_overlap"])  # < size: window_overlap < 1
    out: list[Example] = []
    start, scored = 0, 0  # targets of positions [0, scored) are already scored
    while True:
        end = min(start + size, n)
        keep = scored - start  # leading context positions, targets masked
        out.append(
            (
                ev_in[start:end],
                gap_in[start:end],
                [-1] * keep + ev_out[scored:end],
                [-1] * keep + gap_out[scored:end],
            )
        )
        scored = end
        if end == n:
            return out
        start = end - overlap


def _batches(examples: list[Example], size: int) -> list[list[Example]]:
    return [examples[i : i + size] for i in range(0, len(examples), size)]


def _tensors(torch: Any, batch: list[Example], device: Any) -> tuple[Any, ...]:
    longest = max(len(ev) for ev, _, _, _ in batch)
    shape = (len(batch), longest)
    ev_in = torch.full(shape, PAD, dtype=torch.long)
    gap_in = torch.full(shape, PAD, dtype=torch.long)
    ev_out = torch.full(shape, -1, dtype=torch.long)
    gap_out = torch.full(shape, -1, dtype=torch.long)
    for row, (a, b, c, d) in enumerate(batch):
        ev_in[row, : len(a)] = torch.tensor(a, dtype=torch.long)
        gap_in[row, : len(b)] = torch.tensor(b, dtype=torch.long)
        ev_out[row, : len(c)] = torch.tensor(c, dtype=torch.long)
        gap_out[row, : len(d)] = torch.tensor(d, dtype=torch.long)
    return tuple(t.to(device) for t in (ev_in, gap_in, ev_out, gap_out))


def _loss(
    torch: Any, net: Any, batch: list[Example], hp: Mapping[str, Any], device: Any
) -> Any:
    ev_in, gap_in, ev_out, gap_out = _tensors(torch, batch, device)
    hidden = _forward(torch, net, ev_in, gap_in, ev_in == PAD)
    functional = torch.nn.functional
    event_logits = net["event_head"](hidden)
    l_event = functional.cross_entropy(
        event_logits.reshape(-1, event_logits.shape[-1]),
        ev_out.reshape(-1),
        ignore_index=-1,
    )
    loss = l_event
    if bool((gap_out >= 0).any()):
        time_logits = net["time_head"](hidden)
        l_time = functional.cross_entropy(
            time_logits.reshape(-1, time_logits.shape[-1]),
            gap_out.reshape(-1),
            ignore_index=-1,
        )
        loss = loss + hp["time_loss_weight"] * l_time
    return loss


def _mean_loss(
    torch: Any, net: Any, examples: list[Example], hp: Mapping[str, Any], device: Any
) -> float:
    total, weight = 0.0, 0
    for batch in _batches(examples, hp["batch_size"]):
        n = sum(sum(1 for t in c if t >= 0) for _, _, c, _ in batch)
        total += float(_loss(torch, net, batch, hp, device)) * n
        weight += n
    return total / max(weight, 1)


# ---------------------------------------------------------------------------
# network
# ---------------------------------------------------------------------------


def _network(torch: Any, n_events: int, n_bins: int, hp: Mapping[str, Any]) -> Any:
    """Modules of the network (a ModuleDict, so type checks do not need torch)."""
    nn = torch.nn
    d = hp["d_model"]
    blocks = nn.ModuleList(
        [
            nn.ModuleDict(
                {
                    "ln1": nn.LayerNorm(d),
                    "qkv": nn.Linear(d, 3 * d),
                    "proj": nn.Linear(d, d),
                    "ln2": nn.LayerNorm(d),
                    "ff1": nn.Linear(d, hp["ff_dim"]),
                    "ff2": nn.Linear(hp["ff_dim"], d),
                }
            )
            for _ in range(hp["num_layers"])
        ]
    )
    return nn.ModuleDict(
        {
            "event_embed": nn.Embedding(n_events, d, padding_idx=PAD),
            "gap_embed": nn.Embedding(n_bins + _FIRST_BIN, d, padding_idx=PAD),
            "position": nn.Embedding(hp["max_seq_len"], d),
            "dropout": nn.Dropout(hp["dropout"]),
            "blocks": blocks,
            "ln_final": nn.LayerNorm(d),
            "event_head": nn.Linear(d, n_events),
            "time_head": nn.Linear(d, n_bins),
            "_meta": _Meta(torch, hp["num_heads"]),
        }
    )


def _Meta(torch: Any, heads: int) -> Any:  # noqa: N802 - module factory
    """A parameter-free module carrying the number of attention heads."""
    module = torch.nn.Identity()
    module.heads = heads
    return module


def _attention(
    torch: Any, block: Any, x: Any, heads: int, mask: Any, cache: Any
) -> tuple[Any, Any]:
    """Causal multi-head self-attention; ``cache`` holds past keys/values."""
    batch, steps, d = x.shape
    q, k, v = block["qkv"](x).split(d, dim=-1)

    def heads_first(t: Any) -> Any:
        return t.view(batch, -1, heads, d // heads).transpose(1, 2)

    q, k, v = heads_first(q), heads_first(k), heads_first(v)
    if cache is not None:
        k = torch.cat([cache[0], k], dim=2)
        v = torch.cat([cache[1], v], dim=2)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d // heads)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    out = (weights @ v).transpose(1, 2).reshape(batch, steps, d)
    return block["proj"](out), (k, v)


def _forward(
    torch: Any,
    net: Any,
    events: Any,
    gaps: Any,
    padding: Any,
    cache: list[Any] | None = None,
    offset: int = 0,
) -> Any:
    """Hidden states for ``events``/``gaps`` (batch x steps).

    ``padding`` marks padded positions (masked as keys). With ``cache``
    (incremental decoding), ``offset`` is the number of cached positions and
    the new keys/values are appended to ``cache`` in place.
    """
    steps = events.shape[1]
    positions = torch.arange(offset, offset + steps, device=events.device)
    x = net["event_embed"](events) + net["gap_embed"](gaps) + net["position"](positions)
    x = net["dropout"](x)
    total = offset + steps
    causal = torch.ones(steps, total, dtype=torch.bool, device=events.device).triu(
        offset + 1
    )
    mask = causal[None, None]
    if padding is not None:
        mask = mask | padding[:, None, None, :]
    heads = net["_meta"].heads
    for layer, block in enumerate(net["blocks"]):
        past = cache[layer] if cache is not None and layer < len(cache) else None
        attended, kv = _attention(torch, block, block["ln1"](x), heads, mask, past)
        if cache is not None:
            if layer < len(cache):
                cache[layer] = kv
            else:
                cache.append(kv)
        x = x + net["dropout"](attended)
        hidden = block["ff2"](torch.nn.functional.gelu(block["ff1"](block["ln2"](x))))
        x = x + net["dropout"](hidden)
    return net["ln_final"](x)


def _step(
    torch: Any, net: Any, events: list[int], gaps: list[int], cache: list[Any] | None
) -> tuple[Any, list[Any]]:
    """Hidden state of the last position, reusing ``cache`` when given."""
    if cache is None:
        cache = []
        ev = torch.tensor([events], dtype=torch.long)
        gp = torch.tensor([gaps], dtype=torch.long)
        hidden = _forward(torch, net, ev, gp, None, cache, 0)
    else:
        ev = torch.tensor([[events[-1]]], dtype=torch.long)
        gp = torch.tensor([[gaps[-1]]], dtype=torch.long)
        hidden = _forward(torch, net, ev, gp, None, cache, len(events) - 1)
    return hidden[0, -1], cache


def _weights(torch: Any, logits: Any, temperature: float, top_k: int) -> list[float]:
    """Sampling weights after temperature scaling and optional top-k."""
    scaled = logits / temperature
    if 0 < top_k < scaled.shape[-1]:
        cutoff = torch.topk(scaled, top_k).values[-1]
        scaled = scaled.masked_fill(scaled < cutoff, float("-inf"))
    return [float(w) for w in torch.softmax(scaled, dim=-1).tolist()]
