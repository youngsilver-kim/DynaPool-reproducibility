# Result provenance

The retained A100 evidence records `git_commit: null` because the Colab runs
were launched from an exported source ZIP without the repository's `.git`
directory. The metadata must not be edited retroactively.

During the evidence audit in the original development repository, the archived
source snapshot was compared file by file with commit:

```text
1a5a1991ebac23693f6cf3fc133bcb273c1759e8
```

The tracked experiment files were byte-identical. This clean reproducibility
repository has an independent Git history, so that historical identifier is
provenance information rather than a commit expected to exist here. Later
commits may add paper plotting scripts, public result subsets, and
documentation. They must not be described as the commit that trained the
retained models unless the models are retrained from that later commit.

The full evidence archive contains the source snapshot, run configurations,
50-epoch histories, validation predictions, gate coefficients, software and
hardware metadata, raw latency trials, and the automated audit report.
