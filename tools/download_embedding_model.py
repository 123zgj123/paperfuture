"""Download a pinned public embedding checkpoint with verified file hashes.

All large and temporary files live below the explicitly supplied data directory.
No Hugging Face token, remote Python code, or inference API is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request


def download(repo: str, revision: str, destination: Path) -> dict:
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("revision must be an immutable full commit SHA")
    origin = "https://huggingface.co/"
    with urllib.request.urlopen(origin + "api/models/" + repo + "/revision/" + revision + "?blobs=true", timeout=60) as stream:
        metadata = json.load(stream)
    if metadata["sha"] != revision:
        raise ValueError("server returned a different revision")
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for item in metadata["siblings"]:
        relative = Path(item["rfilename"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("invalid model artifact path")
        if relative.suffix not in (".json", ".safetensors", ".md") and relative.name not in ("LICENSE", "NOTICE"):
            continue
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".part")
        if not path.exists():
            url = origin + repo + "/resolve/" + revision + "/" + urllib.parse.quote(relative.as_posix())
            print("Downloading", relative, item.get("size"), flush=True)
            with urllib.request.urlopen(url, timeout=180) as response, temporary.open("wb") as output:
                while chunk := response.read(4 * 1024 * 1024):
                    output.write(chunk)
            candidate = temporary
        else:
            candidate = path
        sha256 = hashlib.sha256()
        size = candidate.stat().st_size
        blob = hashlib.sha1(("blob " + str(size) + "\0").encode())
        with candidate.open("rb") as stream:
            while chunk := stream.read(4 * 1024 * 1024):
                sha256.update(chunk)
                blob.update(chunk)
        expected = (item.get("lfs") or {}).get("sha256")
        if item.get("size") is not None and item["size"] != size:
            raise ValueError("file size mismatch: " + str(relative))
        if expected and expected != sha256.hexdigest():
            raise ValueError("LFS SHA256 mismatch: " + str(relative))
        if not expected and item.get("blobId") and item["blobId"] != blob.hexdigest():
            raise ValueError("Git blob hash mismatch: " + str(relative))
        if candidate == temporary:
            temporary.replace(path)
        records.append({"path": relative.as_posix(), "bytes": size, "sha256": sha256.hexdigest(),
                        "upstream_hash_verified": bool(expected or item.get("blobId"))})
    manifest = {"repo_id": repo, "revision": revision, "source": origin + repo + "/tree/" + revision,
                "files": records, "total_bytes": sum(r["bytes"] for r in records)}
    temporary = destination / "download-manifest.json.part"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(destination / "download-manifest.json")
    print(json.dumps({"repo_id": repo, "revision": revision, "files": len(records), "bytes": manifest["total_bytes"]}))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    download(args.repo, args.revision, args.destination)
