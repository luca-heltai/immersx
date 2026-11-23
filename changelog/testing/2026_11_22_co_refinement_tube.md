PR: #494

- `LocalCoRefinement.ThicknessRefinesAroundTheTube` checks that post-refinement passes refine at the thickness distance from a segment and not on it. `TensorProductSpace.SerialCellThicknessFollowsRefinedFields` checks the thickness range given to the preprocessing hook after refinement.
