"""Download the pinned training data, evaluation scenes and PI05 initialization."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request

from echo.io import atomic_json, inside, read_json, sha256


def unpack_assets(directory: str | Path) -> None:
    """Expose the released scene pairs under the configured scene root.

    ``LIBERO_Recovery_Assets`` stores the curated 880-scene release as an LFS
    archive whose members are rooted at ``scenes/``.  Evaluation expects the
    suite directories directly below ``data/scenes``.  Extract only regular
    files, strip that archive prefix, and refuse unsafe or conflicting paths.
    """
    directory = Path(directory)
    archive_path = directory / "scenes.tar.gz"
    if not archive_path.is_file():
        raise FileNotFoundError(f"Missing released scene archive: {archive_path}")
    suites = {"selected_bddl_scenes_" + name for name in
              ("spatial", "object", "goal", "10", "90")}
    with tarfile.open(archive_path, "r:gz") as archive:
        members = {}
        for member in archive.getmembers():
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts or path.parts[:1] != ("scenes",):
                raise ValueError("Scene archive contains an unsafe path or unexpected root")
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError("Scene archive contains a non-regular entry")
            relative = Path(*path.parts[1:])
            if (len(relative.parts) < 2 or relative.parts[0] not in suites or
                    not (relative.suffix == ".bddl" or relative.name.endswith("_sim_state.npy")) or
                    relative in members):
                raise ValueError("Scene archive contains an unexpected or duplicate file")
            members[relative] = member
        bddls = {path for path in members if path.suffix == ".bddl"}
        states = {path.with_name(path.stem + "_sim_state.npy") for path in bddls}
        if ({path.parts[0] for path in bddls} != suites or
                set(members) != bddls | states):
            raise ValueError("Scene archive is missing suite data or BDDL/state pairs")
        for relative, member in members.items():
            destination = inside(directory, relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError("Scene archive entry has no file payload")
            temporary = None
            try:
                with stream, tempfile.NamedTemporaryFile(
                        dir=destination.parent, prefix=".scene-", delete=False) as output:
                    temporary = Path(output.name)
                    shutil.copyfileobj(stream, output)
                if destination.exists():
                    if sha256(destination) != sha256(temporary):
                        raise ValueError(f"Refuse replacing a different scene file: {destination}")
                else:
                    temporary.replace(destination)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)


def fetch_source(source):
    target = Path(source["local_dir"])
    provider, repository, revision = source["provider"], source["repo_id"], source["revision"]
    if not repository or not revision:
        raise ValueError("This input has no published repository/revision; configure the real anonymous release first")
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Pin a complete 40-character revision, not main/HEAD")
    if provider == "huggingface":
        from huggingface_hub import snapshot_download
        snapshot_download(repository, repo_type="dataset", revision=revision, local_dir=target)
        if source.get("format") == "echo_failure_shards_v1":
            from echo.data.failure_archive import unpack_failure
            unpack_failure(target)
    elif provider == "modelscope":
        if not (target/".git").exists():
            if target.exists() and any(target.iterdir()):
                raise FileExistsError("Refuse cloning into a nonempty directory")
            target.parent.mkdir(parents=True, exist_ok=True)
            env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
            subprocess.run(["git", "clone", "https://www.modelscope.cn/datasets/"+repository+".git", str(target)],
                           env=env, check=True)
        subprocess.run(["git", "-C", str(target), "checkout", "--detach", revision], check=True)
        actual = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
        if actual != revision:
            raise ValueError("ModelScope revision differs")
        subprocess.run(["git", "-C", str(target), "lfs", "install", "--local"], check=True)
        subprocess.run(["git", "-C", str(target), "lfs", "pull"], check=True)
    else:
        raise ValueError("Unknown provider")
    atomic_json(target/"download_receipt.json", {"provider": provider, "repo_id": repository, "revision": revision})


def fetch_weights(config):
    from huggingface_hub import snapshot_download
    weights = config["weights"]
    snapshot_download(weights["repo_id"], revision=weights["revision"], local_dir=weights["local_dir"],
                      allow_patterns=[weights["subdir"]+"/fp32/model.safetensors",
                                      weights["subdir"]+"/config.yaml", weights["subdir"]+"/dataset_statistics.json"])
    model = Path(weights["local_dir"])/weights["subdir"]/"fp32/model.safetensors"
    if sha256(model) != weights["sha256"]:
        raise ValueError("Public PI05 weight checksum differs")
    tokenizer = config["tokenizer"]
    target = Path(tokenizer["local_file"])
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        temporary = target.with_suffix(".download")
        with urllib.request.urlopen(tokenizer["url"], timeout=60) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out)
        if sha256(temporary) != tokenizer["sha256"]:
            raise ValueError("Tokenizer checksum differs")
        os.replace(temporary, target)
    if sha256(target) != tokenizer["sha256"]:
        raise ValueError("Tokenizer checksum differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.json")
    parser.add_argument("--source", choices=["success", "success90", "recovery", "failure", "assets", "weights", "all"], default="all")
    args = parser.parse_args()
    config = read_json(args.config)
    names = [*config["sources"], "assets", "weights"] if args.source == "all" else [args.source]
    for name in names:
        if name == "weights":
            fetch_weights(config)
        else:
            source = config["assets"] if name == "assets" else config["sources"][name]
            fetch_source(source)
            if name == "assets":
                unpack_assets(source["local_dir"])


if __name__ == "__main__":
    main()
