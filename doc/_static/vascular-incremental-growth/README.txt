Frozen native SI point-demand experiment, source 5ed0570.
The subsequent b9a10db change only adds an explicit Boost equality include.

measurements.csv: each successful point-demand case, historical endpoint cases,
operation/phase/communication counts and native rank memory statistics.
summary.json: serial three-seed median/min/max.
manifest.json: source/binary/runner/archive hashes and figure provenance.
av-audit.json: stopped AV campaign, inherited failure and diagnostic scope.
render-report.py: exact renderer used to validate archives and export figures.

On the original worker the immutable case archives are under:
/data/home/lheltai/.codex-hosts/lnx3/visualizations/2026/10/29/01a18880-b60d-70f3-8904-1b7d6efb768d/vascular-build-storage
The five campaign directory names are recorded in manifest.json.
Each archive includes runner.py, bin/, input.prm, case.json, physical CSV tables
and MPI ledgers. The historical manifest verification is recorded in
baseline-verification.json (included here).

The renderer uses Python 3, matplotlib 3.10.9 and the checked-in native table
readers. Reproduction takes --source REPOSITORY --storage ARCHIVE_PARENT
--baseline VERIFIED_BASELINE_JSON --output NEW_OUTPUT_DIRECTORY. It validates
inputs/binaries/physical tables before plotting; it does not run a generator.
Use a fresh output directory. Initialization, ordinary and finalization counts
must remain separate. Profile bytes are logical payloads, not wire traffic.

To relocate archived results, update only their manifest paths in a copy of
baseline-verification.json; preserve the original manifests and their hashes.
