"""Exact stochastic continuation and failure propagation across CPU ranks."""
from datetime import timedelta
import multiprocessing as mp
from pathlib import Path
import random
import tempfile
import time
import unittest

import numpy as np
from omegaconf import OmegaConf
import torch
import torch.distributed as dist

from echo.training import checkpoint
from echo.training.distributed import all_rank_call, rank_zero_call
from echo.io import atomic_json, read_json


def tiny_model():
    model = torch.nn.Linear(3, 2)
    model.config = OmegaConf.create({"framework": {"name": "PI05"}})
    return model


def identity_at(root):
    source = Path(root)/"source"
    (source/"fp32").mkdir(parents=True, exist_ok=True)
    atomic_json(source/"dataset_statistics.json", {})
    return {"config": {"method": "echo", "checkpoint": str(source/"fp32/model.safetensors")}}


def updates(model, optimizer, generator, count):
    for _ in range(count):
        inputs = torch.randn(4, 3)+float(np.random.random())+random.random()
        target = torch.randn(4, 2, generator=generator)
        optimizer.zero_grad(set_to_none=True)
        (model(inputs)-target).square().mean().backward()
        optimizer.step()


def distributed_worker(rank, root):
    # On network filesystems, the first ZeRO import can outlast the actual
    # collective test. Bound startup separately from communication/I/O.
    from torch.distributed.optim import ZeroRedundancyOptimizer
    (Path(root)/f"ready_{rank}").touch()
    dist.init_process_group("gloo", init_method="file://"+str(Path(root)/"rendezvous"),
                            rank=rank, world_size=2, timeout=timedelta(seconds=25))
    failures = []
    try:
        def fail():
            raise OSError("simulated disk failure")
        for operation in (lambda: rank_zero_call(fail, 2),
                          lambda: all_rank_call(fail if rank == 1 else lambda: None, 2)):
            try:
                operation()
            except RuntimeError as error:
                failures.append(str(error))
        # Exercise checkpoint prepare and actual ZeRO optimizer partitions.
        torch.manual_seed(42)
        model = tiny_model()
        optimizer = ZeroRedundancyOptimizer(model.parameters(), optimizer_class=torch.optim.AdamW, lr=.01)
        identity = identity_at(root) if rank == 0 else None
        payload = [identity]; dist.broadcast_object_list(payload, 0); identity = payload[0]
        run = Path(root)/"distributed"
        if rank == 0:
            (run/"checkpoints/step_00000001").mkdir(parents=True)
        dist.barrier()
        try:
            checkpoint.save(run, model, optimizer, 1, {"epoch": 0, "batch_offset": 1}, identity,
                            rank, 2, torch.Generator())
        except RuntimeError as error:
            failures.append(str(error))
        optimizer.zero_grad(set_to_none=True)
        model(torch.ones(2, 3)).sum().backward(); optimizer.step()
        directory = checkpoint.save(run, model, optimizer, 2, {"epoch": 0, "batch_offset": 2}, identity,
                                    rank, 2, torch.Generator())
        recovered = checkpoint.resume(directory, model, optimizer, identity, rank, 2)
        atomic_json(Path(root)/f"rank_result_{rank}.json", {"failures": failures, "step": recovered[0]})
    finally:
        dist.destroy_process_group()


class CheckpointContracts(unittest.TestCase):
    def test_stochastic_resume_matches_continuous_updates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            random.seed(42); np.random.seed(42); torch.manual_seed(42)
            model = tiny_model(); optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
            generator = torch.Generator().manual_seed(31)
            updates(model, optimizer, generator, 3)
            identity = identity_at(root)
            directory = checkpoint.save(root, model, optimizer, 3, {"epoch": 1, "batch_offset": 2},
                                        identity, 0, 1, generator)
            updates(model, optimizer, generator, 3)
            expected = {key: value.clone() for key, value in model.state_dict().items()}
            resumed = tiny_model(); resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=.01)
            step, cursor, rng, loader_rng = checkpoint.resume(directory, resumed, resumed_optimizer, identity, 0, 1)
            self.assertEqual(step, 3); self.assertEqual(cursor, {"epoch": 1, "batch_offset": 2})
            generator.set_state(loader_rng); checkpoint.restore_rng(rng)
            updates(resumed, resumed_optimizer, generator, 3)
            for key, value in resumed.state_dict().items():
                torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
            for a, b in zip(optimizer.state.values(), resumed_optimizer.state.values()):
                for key in a:
                    torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
            # Unexpected files and altered rank states must be rejected.
            (directory/"extra").write_text("unexpected")
            with self.assertRaises(ValueError): checkpoint.verify(directory)
            (directory/"extra").unlink()
            with (directory/"rank_00.pt").open("ab") as handle: handle.write(b"changed")
            with self.assertRaises(ValueError): checkpoint.verify(directory)

    def test_two_rank_checkpoint_and_io_errors_do_not_hang(self):
        with tempfile.TemporaryDirectory() as temporary:
            context = mp.get_context("spawn")
            processes = [context.Process(target=distributed_worker, args=(rank, temporary)) for rank in range(2)]
            for process in processes: process.start()
            try:
                startup_deadline = time.monotonic()+180
                while not all((Path(temporary)/f"ready_{rank}").exists() for rank in range(2)):
                    self.assertTrue(all(p.is_alive() or p.exitcode == 0 for p in processes),
                                    "Worker failed during imports")
                    self.assertLess(time.monotonic(), startup_deadline, "Worker imports exceeded startup timeout")
                    time.sleep(.1)
                for process in processes:
                    process.join(40)
                    self.assertFalse(process.is_alive(), "Collective hung after a rank's disk error")
                    self.assertEqual(process.exitcode, 0)
                for rank in range(2):
                    row = read_json(Path(temporary)/f"rank_result_{rank}.json")
                    self.assertEqual(len(row["failures"]), 3)
                    self.assertEqual(row["step"], 2)
            finally:
                for process in processes:
                    if process.is_alive(): process.terminate(); process.join(5)


if __name__ == "__main__":
    unittest.main()
