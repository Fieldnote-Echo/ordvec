# ordvec-manifest

Repo-local, publish=false sidecar verifier for ordvec index manifests.

It verifies index bytes, probed header metadata, row identity, and attestation
shape before a caller loads an ordvec index. It does not sign artifacts, manage
keys, call networks, mutate index files, decide deployment trust policy, or
change the C ABI.

```sh
cargo run -p ordvec-manifest -- create \
  --index path/to/index.tvrq \
  --row-id-is-identity \
  --embedding-model bge-small-en-v1.5 \
  --out path/to/index.manifest.json

cargo run -p ordvec-manifest -- verify --manifest path/to/index.manifest.json
```

The schema version is `ordvec.index_manifest.v1`. Relative paths resolve from
the manifest file's directory, absolute paths are rejected by default, and
relative paths may not escape the manifest directory unless explicitly allowed.
