# Experiment-only release scope

This is a newly filtered experiment-only payload, built independently of the earlier
unpublished packaging candidates. It excludes all submission text, formatted documents,
publisher templates, author metadata, writing/build scripts and manuscript figure folders.
Only code, experimental inputs, complete measured runs, analyses and dependency notices
are included. Earlier draft-package archives must not be substituted for this asset.

The original 27 frozen scientific files, model manifests, numerical CSV records and
483 declared output images retain their delivered bytes. No GPU experiment was rerun.
Private editor URLs and local personal paths were already removed from non-frozen
metadata; those removals do not change the scientific records. The new weight helper
writes download verification separately without modifying historical manifests.

The code-only checkout flattens small statistics under results/ for browsing; the full
experiment asset retains original paths. See results/INDEX.md where supplied.
PUBLIC_RELEASE_MANIFEST.json hashes this payload and excludes itself.
