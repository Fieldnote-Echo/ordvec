# Security Policy

## Supported versions

`ordvec` is pre-1.0; security fixes land on `main` and the latest published
release.

## Reporting a vulnerability

Please report security issues **privately** — do not open a public issue.

Use GitHub's private vulnerability reporting:
**Security → Report a vulnerability**
(<https://github.com/Fieldnote-Echo/ordvec/security/advisories/new>).

We aim to acknowledge reports within a few business days.

`ordvec` parses serialized index files (`.tvr` / `.tvrq` / `.tvbm` /
`.tvsb`); the loaders are fuzzed (`cargo +nightly fuzz`), so
parsing-robustness reports against the deserialization paths are especially
welcome.
