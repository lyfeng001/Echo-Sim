"""Small collectives that propagate Python I/O errors to every rank."""
from __future__ import annotations


def rank_zero_call(operation, world_size):
    import torch.distributed as dist

    rank = dist.get_rank() if world_size > 1 else 0
    payload = [None, None]
    if rank == 0:
        try:
            payload[0] = operation()
        except Exception as error:
            payload[1] = f"{type(error).__name__}: {error}"
    if world_size > 1:
        dist.broadcast_object_list(payload, src=0)
    if payload[1] is not None:
        raise RuntimeError("Rank 0 operation failed: " + payload[1])
    return payload[0]


def all_rank_call(operation, world_size):
    import torch.distributed as dist

    value, error = None, None
    try:
        value = operation()
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"
    errors = [error]
    if world_size > 1:
        errors = [None] * world_size
        dist.all_gather_object(errors, error)
    failures = [f"rank {i}: {e}" for i, e in enumerate(errors) if e is not None]
    if failures:
        raise RuntimeError("Distributed operation failed: " + "; ".join(failures))
    return value
