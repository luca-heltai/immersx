PR: pending

- The coupled MetricFlowX state Jacobian now retains its immutable assembled matrix owner. Reassembling at a new nonlinear state detaches the matrix, so older linearization snapshots remain valid without cloning the full sparse matrix for each snapshot.
