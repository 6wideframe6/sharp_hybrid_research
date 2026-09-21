"""Read-only verification against the existing frozen research manifest."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    manifest_path = ROOT / "results/tinyvim_local/frozen_baseline_hashes.json"
    manifest = json.loads(manifest_path.read_text())
    failures = []
    total_bytes = 0
    for name, expected in manifest.items():
        path = ROOT / name
        if not path.is_file():
            failures.append({"path": name, "reason": "missing"})
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
                total_bytes += len(chunk)
        if digest.hexdigest() != expected:
            failures.append({"path": name, "reason": "hash mismatch"})
    result = {"manifest": str(manifest_path.relative_to(ROOT)),
              "files_checked": len(manifest), "bytes_checked": total_bytes,
              "passed": not failures, "failures": failures}
    destination = Path(__file__).resolve().parent / "preservation.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
