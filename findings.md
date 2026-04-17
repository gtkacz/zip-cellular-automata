# Findings

## Source paper
Chatzinikolaou, T. P., Karamani, R.-E., Fyrigos, I.-A., & Sirakoulis, G. Ch. (2024). *Handling Sudoku puzzles with irregular learning cellular automata*. Natural Computing, 23(1), 41–60. https://doi.org/10.1007/s11047-024-09975-4

### Key elements ported from the paper
- **Cell as Learning Automaton (LA):** action probability vector `P_{i,j}` over an action set `T_{i,j}`.
- **Reward/penalty rule:** multiplicative — `P_{i,j}^{t+1}(c) = P_{i,j}^t(c) · r(d)` (reward, r > 1) or `· p(d)` (penalty, p < 1). See Algorithm 1 in paper.
- **Cell degree** `d_{i,j}`: # of neighbors that *differ* from this cell's chosen action.
- **Iteration until convergence**: continue until `FinalizedCells == TotalCells`.
- **Enhancements** (Section 5 of paper, considered for future): variable learning rates; selective probability reset rule.

### Why the paper's mechanism doesn't trivially port to Zip
- Sudoku constraints are **purely local** ("differ from 20 neighbors"). Local degree → reward.
- Zip constraints are **global** (single Hamiltonian path through ordered waypoints). Local degree alone cannot detect connectivity violations or cycles.
- Solution: introduce a *second CA layer* (chemical diffusion field) that converts global topology into a locally-readable signal.

## Cross-domain analogy
*Physarum polycephalum* (slime mold) solves shortest-path and network-design problems via reaction-diffusion-like reinforcement of high-flux channels:
- Tero, A., Takagi, S., Saigusa, T., Ito, K., Bebber, D. P., Fricker, M. D., Yumiki, K., Kobayashi, R., & Nakagaki, T. (2010). *Rules for biologically inspired adaptive network design*. Science, 327(5964), 439–442.
- Nakagaki, T., Yamada, H., & Tóth, Á. (2000). *Maze-solving by an amoeboid organism*. Nature, 407, 470.

Our two-layer CA is a discrete digital analogue: chemical concentrations "carve" channels along candidate paths; cells reinforce shape-choices that lie along those channels.

## Failure modes anticipated (without mitigation)
1. **Symmetric local minima**: tied chemical fields freeze the system → mitigated by per-cell stochastic noise.
2. **Disconnected high-concentration islands** from bad early commitments → mitigated by selective probability reset (paper §5) — deferred to phase 6.
3. **Closed loops** unconnected to the main path → caught by terminal trace check; restart with new init.

## Pure-CA constraint
Diffusion is itself local averaging, so both layers remain pure CA. Per-cell RNG noise is local (each cell decides for itself), so it preserves "no central coordinator." A trace-from-#1 connectivity check at termination is *external verification*, not part of the CA dynamics — it only decides "stop or restart."
