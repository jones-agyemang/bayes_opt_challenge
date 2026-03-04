# Week 5

Adjustments Log
- 
  - Currently using a gradient-based minimizer with 10 multi-starts. Switching to randomly sampling sampling a larger space and submitting the best candidate after evaluating all `n` candidate. This change ensures that the GP generates `n` number of possible candidates(`X_next`) in the given search space $[0, 1]^d$. The acquisition point is evaluated at the generated candidate points, with the highest acquisition value being chosen as the final value
  This is computationally intensive but, it should help to reach our maximisation goal faster. 

## Tunable Parameters
||||