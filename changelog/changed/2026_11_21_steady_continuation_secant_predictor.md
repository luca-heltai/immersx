PR: #485

- Seed steady residual-continuation stages with a secant predictor from the two latest accepted states, falling back to the latest accepted state if the prediction is inadmissible.
