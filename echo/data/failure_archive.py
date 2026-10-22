"""Verify and extract the published terminal-failure shards."""
from __future__ import annotations

import hashlib
from pathlib import Path
import tarfile

from echo.io import inside, read_json, sha256


def verify_archive(path, inventory, expected_members=None, output=None):
    """Stream/hash allowlisted regular files; never use tar.extractall."""
    seen = set()
    with tarfile.open(path, "r:") as archive:
        for member in archive:
            name = member.name
            if (not member.isfile() or name not in inventory or name in seen or
                    Path(name).is_absolute() or ".." in Path(name).parts):
                raise ValueError("Shard contains an unexpected, duplicate or unsafe entry")
            record = inventory[name]
            if member.size != record["bytes"] or member.uid or member.gid or member.uname or member.gname or member.mtime:
                raise ValueError("Shard size or anonymous header differs")
            stream = archive.extractfile(member)
            hasher = hashlib.sha256()
            destination = inside(output, name) if output is not None else None
            temporary = destination.with_name(destination.name+".pending") if destination else None
            if destination:
                destination.parent.mkdir(parents=True, exist_ok=True)
            writer = temporary.open("wb") if temporary else None
            try:
                for block in iter(lambda: stream.read(1024*1024), b""):
                    hasher.update(block)
                    if writer:
                        writer.write(block)
            finally:
                stream.close()
                if writer:
                    writer.close()
            if hasher.hexdigest() != record["sha256"]:
                if temporary:
                    temporary.unlink(missing_ok=True)
                raise ValueError("Shard object checksum differs")
            if destination:
                if destination.exists() and sha256(destination) != record["sha256"]:
                    temporary.unlink()
                    raise ValueError("Refuse overwriting a different local failure object")
                temporary.replace(destination)
            seen.add(name)
    if expected_members is not None and seen != set(expected_members):
        raise ValueError("Shard member set differs")
    return seen


def unpack_failure(directory):
    """Validate the complete downloaded package before exposing its trajectories."""
    directory = Path(directory)
    manifest = read_json(directory/"archives.json")
    if manifest["schema_version"] != 1 or manifest["format"] != "echo_failure_shards_v1":
        raise ValueError("Unknown failure distribution format")
    if sha256(directory/"inventory.json") != manifest["inventory_sha256"]:
        raise ValueError("Failure inventory checksum differs")
    inventory, seen = read_json(directory/"inventory.json"), set()
    for record in manifest["archives"]:
        path = inside(directory, record["path"])
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError("Downloaded shard checksum differs")
        members = verify_archive(path, inventory, output=directory)
        if seen & members or len(members) != record["objects"]:
            raise ValueError("Duplicate or incomplete shard contents")
        seen.update(members)
    if seen != set(inventory) or len(seen) != manifest["objects"]:
        raise ValueError("Incomplete failure dataset package")
    return manifest
