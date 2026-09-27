# Public release manifest

**Release:** source-only reset, 2026-09-27

## Inclusion rule

The release retains root documentation, dependency information, first-party
scripts, source-only Kaggle recipes, two zero-output notebooks, and scalar
audit JSON whose reviewed structure contains no
record-level competition content. The retained report files describe model or
adapter structure only.

The recipe batch is a method/provenance release: it includes source and
dependency references, but no downloaded dependencies or execution products.
`recipes/README.md` identifies the included files and the intentionally
omitted generated bundle.

## Exclusion rule

The release excludes all organizer data; literal prompts; IDs; answers;
predictions; generated completions; token/context traces; handoff copies;
external notebook copies; copied upstream checkpoints/adapters/submissions;
caches; and local paths. The ignore rules encode the main generated-output
families so they cannot be recommitted accidentally.

This source-only repository still does not track model binaries. Eligible
first-party checkpoints, adapters, compiled model artifacts and submission
bundles are published separately with source revision, provenance and checksum
records in the linked artifact dataset described by `ARTIFACTS.md`.

## History cleanup

The earlier public branch contained record-level research exports. It was
replaced with a new source-only root commit so the excluded files are no longer
reachable from the public default branch. A public history rewrite cannot
retract copies already cloned by others; it does remove the material from the
repository's visible branch and future clones.

## Verification required before publish

1. Scan tracked structured reports for prohibited record-level fields.
2. Reject unexpected large files, archives, weights, and local absolute paths.
3. Run `python3 -m compileall scripts`.
4. Read back the pushed default-branch commit and clone it into a clean
   directory.
