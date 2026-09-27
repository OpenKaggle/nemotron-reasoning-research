# Artifact boundary

The local research workspace contains adapters, submission bundles, generated
traces, and third-party downloads. They are excluded from Git because their
rights, provenance, or size make a public source repository the wrong delivery
mechanism. That does not make every user-authored artifact private: eligible
models and submission bundles use a documented public artifact host instead.

For every artifact considered for a future release, record:

1. origin, author, license, and competition-rule status;
2. whether it is source, a user-authored derivative, or a third-party binary;
3. SHA-256 and size;
4. a permitted retrieval location; and
5. a clean restore check.

Only a separately reviewed, authorized artifact may be placed on an external
host. The first public artifact release is the [OpenKaggle Nemotron Model and
Submission Artifacts](https://www.kaggle.com/datasets/jahyee/openkaggle-nemotron-model-submission-artifacts)
dataset: it holds one first-party LoRA submission archive and compact patches
for user-authored sweep changes, with manifests and checksums. It deliberately
does not mirror base-model bytes, organizer data, or complete third-party
adapters.

For any artifact not yet reviewed, a source link plus reconstruction
instructions remains the public record, and the original remains outside this
repository.
