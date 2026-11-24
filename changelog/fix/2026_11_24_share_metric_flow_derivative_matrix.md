PR: #508

- Share the cached state-independent MetricFlowX derivative matrix across immutable solver snapshots, while retaining copy-on-write versions for later invalidations.
