# Artifact boundary

The local research workspace contains adapters, submission bundles, generated
traces, and third-party downloads. They are excluded from Git because their
rights, provenance, or size make a public source repository the wrong delivery
mechanism.

For every artifact considered for a future release, record:

1. origin, author, license, and competition-rule status;
2. whether it is source, a user-authored derivative, or a third-party binary;
3. SHA-256 and size;
4. a permitted retrieval location; and
5. a clean restore check.

Only a separately reviewed, authorized artifact may be placed on an external
host. Until then, a source link plus reconstruction instructions is the public
record, and the original remains outside this repository.
