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
checkpoints; adapters; submissions; external notebook copies; caches; and local
paths. The ignore rules encode the main generated-output families so they cannot
be recommitted accidentally.

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
