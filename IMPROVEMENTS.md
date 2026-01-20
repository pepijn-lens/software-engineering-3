# Improvements to Increase Crash Likelihood

## Key Changes

### 1. Aggressive Initial Configuration
- `vehicles_count`: Set to 90% of maximum (54 vehicles instead of random)
- `initial_spacing`: Set to minimum (0.5 instead of random)
- `ego_spacing`: Set to minimum (1.0 instead of random)
- `lanes_count`: Set to minimum (3 lanes instead of random)
- `initial_lane_id`: Set to 0 (leftmost lane)

**Rationale**: Start with the most congested, crash-prone scenario possible.

### 2. Deterministic Crash-Biased Mutations
Instead of probabilistic mutations, now:
- **vehicles_count**: ALWAYS increases (never decreases)
  - Step size: 2-8 vehicles per mutation
  - More vehicles = more traffic = higher collision risk
  
- **initial_spacing & ego_spacing**: ALWAYS decreases (never increases)
  - Step size: 0.2-0.8 units per mutation
  - Tighter spacing = vehicles closer together = higher collision risk
  
- **lanes_count**: Decreases when possible
  - Fewer lanes = more congestion = higher collision risk
  
- **initial_lane_id**: Random valid lane
  - Explores different starting positions

**Rationale**: Remove randomness that could lead to "safer" scenarios. Always push towards crashes.

### 3. Crash-Prone Parameter Initialization
When a parameter doesn't exist in the config:
- `vehicles_count`: Initialize to 80% of max
- `initial_spacing`/`ego_spacing`: Initialize to minimum
- `lanes_count`: Initialize to minimum
- `initial_lane_id`: Initialize to 0

**Rationale**: New parameters start in crash-prone state, not neutral.

### 4. Better Fitness Tracking
- Print min_distance in addition to fitness
- Track crashes vs. close calls
- Early stopping when crash is found

## Expected Results

With these changes:
1. **Faster crash discovery**: Aggressive mutations push towards crashes quickly
2. **Smaller distances**: Even without crashes, scenarios should have vehicles closer together
3. **Better convergence**: Deterministic mutations prevent "backtracking" to safer scenarios
4. **More informative output**: See both crash status and distance metrics

## Trade-offs

- **Less exploration**: Deterministic mutations may miss some scenarios
- **Local optima**: May converge to specific crash patterns
- **Mitigation**: Multiple neighbors per iteration (8-10) provides some exploration
