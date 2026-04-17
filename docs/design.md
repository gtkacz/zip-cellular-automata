# Zip Cellular Automata — Design Document

**Status:** Draft v0.1 (2026-04-17)
**Scope:** A dual-layer cellular automaton solver for the LinkedIn *Zip* puzzle, adapted from the Irregular Learning Cellular Automaton (ILCA) framework of Chatzinikolaou et al. (2024).

---

## 1. Problem Statement

A *Zip* puzzle consists of:

- An $N \times N$ grid of cells.
- A subset $\mathcal{W} \subseteq \{1, 2, \dots, N^2\}$ of cells pre-labelled with strictly-increasing positive integers $1, 2, \dots, K$ (the **waypoints**). Let $K = |\mathcal{W}|$.
- An optional set of **wall constraints**: for any cell, zero or more of its four edges may be marked impassable.

A valid solution is a sequence of cells $\pi_1, \pi_2, \dots, \pi_{N^2}$ satisfying:

1. $\pi_1$ is the cell labelled $1$ and $\pi_{N^2}$ is the cell labelled $K$.
2. Consecutive cells $\pi_t, \pi_{t+1}$ are 4-adjacent and the edge between them is not walled.
3. Every cell of the grid appears exactly once (Hamiltonian path).
4. Waypoints are visited in ascending order: if $\pi_a$ has label $k$ and $\pi_b$ has label $k+1$, then $a < b$.
5. The path neither branches nor crosses itself (implied by Hamiltonian).

This is a constrained Hamiltonian-path problem on a grid graph with ordered checkpoints. NP-hard in general; trivial-to-modest at LinkedIn sizes ($N \in [5, 8]$).

### 1.1 Goal of this work

Produce a **pure cellular automaton** that, without any centralised coordinator, finds a valid solution to small Zip puzzles. The contribution is methodological: extending the ILCA technique to a global-constraint problem class via a slime-mold–inspired auxiliary diffusion field.

Solve speed is explicitly *not* a goal. Backtracking dispatches LinkedIn-grade puzzles in microseconds. Our targets are reliability, interpretability of the emergent dynamics, and the side-by-side visual narrative of two coupled CA layers reaching consensus.

---

## 2. Background

### 2.1 The ILCA Sudoku method (Chatzinikolaou et al., 2024)

Each Sudoku cell $C_{i,j}$ is treated as a Learning Automaton holding an action probability vector $P_{i,j} \in \Delta^9$ over the actions $\{1, \dots, 9\}$. At each tick:

$$c_{i,j}^t = \arg\max_n P_{i,j}^t(n)$$

A **degree** $d_{i,j}$ counts neighbours with a different chosen action, where the neighbourhood $N_{i,j}$ is the union of row, column, and $3\times3$ sub-grid less the cell itself ($|N_{i,j}|=20$).

**Reward / penalty rule (their Algorithm 1):**

- If $d_{i,j} = |N_{i,j}|$ (no conflict at all): reward.
- Else if $d_{i,j} > \max\{d_{k,l} : C_{k,l} \in N_{i,j},\ c_{k,l} = c_{i,j}\}$ (least-bad among conflicting peers): reward.
- Else: penalise.

Update is multiplicative: $P_{i,j}^{t+1}(c_{i,j}^t) = P_{i,j}^t(c_{i,j}^t) \cdot r$ or $\cdot p$, with $r > 1 > p > 0$, followed by renormalisation. Iteration stops when every cell has its degree saturated.

### 2.2 Why the rule does not transfer directly

Sudoku constraints are **fully decidable from a cell's neighbourhood**. Zip's win condition — "the cells form a single ordered Hamiltonian path" — is a *global topological* property. Two adjacent cells can each be locally happy with their shape choice while the global path is broken into two disconnected components, contains a small cycle, or violates waypoint ordering.

Any direct port of the ILCA degree-rule will converge on locally-consistent but globally-invalid configurations. We need a mechanism that **transports global information into the local update**, while keeping every operation a local CA rule.

---

## 3. Design Overview

We use a **two-layer cellular automaton** with bidirectional coupling.

**Layer 1 — Path-Shape ILCA.** Each cell is an LA whose action set is a small finite set of *path shapes* (how the path threads through the cell). Choosing a shape commits the cell to opening exactly two of its four edges (or one, for a path endpoint).

**Layer 2 — Chemical Diffusion Field.** Each cell additionally holds $K-1$ scalar concentrations, one per **segment** of the eventual path. Segment $k$ runs from waypoint $k$ to waypoint $k+1$. Each waypoint cell perpetually emits the chemicals of its two adjacent segments. Diffusion proceeds locally — but **only through the open ports of the current Layer-1 shape**. This is the core coupling.

The **reward** for Layer 1 is computed locally from the gradient of Layer 2: a shape is reinforced when its open ports point along the steepest-ascent direction of one segment chemical at one port and the next segment's chemical at the other port. The shape "lies along a chemical channel."

The system is a discrete digital analogue of the slime-mold dynamic studied by Tero et al. (2010, *Science*): channels carry chemicals → high concentration reinforces channels → consensus emerges from positive feedback, with no global controller.

### 3.1 Pure-CA invariants preserved

| Operation | Local? | Notes |
|---|---|---|
| Shape probability update | ✓ | Function of cell's own state + 4 neighbours' chemical concentrations only. |
| Diffusion update | ✓ | Standard 5-point stencil masked by current shape. |
| Noise injection (§7) | ✓ | Each cell rolls its own RNG. |
| **Termination check (trace)** | ✗ | External validator; not part of CA dynamics. |

The trace check is the only non-local operation, and it is purely *observational* — it decides whether to halt or restart, never influencing cell state mid-evolution.

---

## 4. Layer 1 — Path-Shape ILCA

### 4.1 Action set

The eight shape primitives, each defined by which of the four edge-ports {N, E, S, W} are *open*:

| Shape | Open ports | Class |
|---|---|---|
| `H` (─) | E, W | through |
| `V` (│) | N, S | through |
| `NE` (└) | N, E | through |
| `NW` (┘) | N, W | through |
| `SE` (┌) | S, E | through |
| `SW` (┐) | S, W | through |
| `END_N` | N | endpoint |
| `END_E` | E | endpoint |
| `END_S` | S | endpoint |
| `END_W` | W | endpoint |

(10 total. We list 4 endpoint shapes — one per direction the lone open port can face.)

### 4.2 Per-cell action set restriction

Let $\mathcal{S}$ denote the full set above. For cell $(i,j)$, the **allowed action set** $\mathcal{A}_{i,j} \subseteq \mathcal{S}$ is the intersection of three filters:

1. **Wall filter.** Any shape with an open port crossing a walled edge is removed.
2. **Boundary filter.** Any shape with an open port pointing off-grid is removed (equivalent to a wall on the perimeter).
3. **Waypoint filter.**
   - If the cell is waypoint #1 or waypoint #K: only endpoint shapes are allowed.
   - If the cell is any other waypoint ($2 \le k \le K-1$): only through-shapes are allowed (the path enters and exits, so two ports must be open).
   - Otherwise (unnumbered cell): only through-shapes are allowed (every cell is on the path; no skips).

Note unnumbered cells get exactly the 6 through-shapes by construction. Endpoint shapes are reserved for the two terminal waypoints.

### 4.3 Probability vector

Each cell holds $P_{i,j} \in \Delta^{|\mathcal{A}_{i,j}|}$ — a probability distribution over its allowed actions. Initialised uniformly. The chosen shape at tick $t$ is $s_{i,j}^t = \arg\max P_{i,j}^t$.

### 4.4 Update rule

At every tick, after Layer 2 has diffused (§5), each cell:

1. Computes a **score** $\sigma(s)$ for every shape $s \in \mathcal{A}_{i,j}$ — see §6.
2. For each shape $s$, multiplies $P_{i,j}(s)$ by $r$ if $\sigma(s)$ exceeds a threshold $\theta_+$, by $p$ if it falls below $\theta_-$, and leaves it unchanged in the dead band $[\theta_-, \theta_+]$.
3. Renormalises $P_{i,j}$.
4. Re-selects $s_{i,j}^{t+1} = \arg\max P_{i,j}^{t+1}$.

Hyperparameters $r > 1 > p > 0$ are constants of the run (cf. paper §4).

---

## 5. Layer 2 — Chemical Diffusion Field

### 5.1 State

Each cell holds a vector $\mathbf{u}_{i,j} \in \mathbb{R}_{\ge 0}^{K-1}$ of non-negative concentrations, one per segment chemical. Index $k \in \{1, \dots, K-1\}$ refers to segment $k$, the (eventual) path subsequence from waypoint $k$ to waypoint $k+1$.

### 5.2 Sources

Waypoint cell $W_k$ ($1 \le k \le K$) clamps:

- $u_{W_k}[k-1] := C_0$ if $k > 1$
- $u_{W_k}[k] := C_0$ if $k < K$

with $C_0$ a fixed source intensity. Sources are re-asserted *after* the diffusion step every tick (Dirichlet boundary condition).

### 5.3 Shape-gated diffusion

Standard 5-point diffusion fails because chemicals must respect the path topology proposed by Layer 1 — otherwise the field would spill everywhere and gradients would carry no path information.

For each chemical $k$, the next-tick concentration is:

$$
u_{i,j}^{t+1}[k] = (1-\delta) \cdot \left[ (1-\alpha) \cdot u_{i,j}^t[k] + \alpha \cdot \frac{1}{|\Omega_{i,j}^t|} \sum_{(p,q) \in \Omega_{i,j}^t} u_{p,q}^t[k] \right]
$$

where $\Omega_{i,j}^t$ is the set of neighbours $(p,q)$ such that **both** the open-port set of $s_{i,j}^t$ and the open-port set of $s_{p,q}^t$ contain the edge between them. That is, chemicals only flow across a *mutually open* edge. $\alpha \in (0, 1]$ is the diffusion rate; $\delta \in [0, 1)$ the decay (combats unbounded accumulation away from sources). If $|\Omega_{i,j}^t| = 0$, the cell holds (no neighbour-mixing term).

This shape-gated Laplacian is the unique non-trivial coupling: chemicals see the same topology Layer 1 currently proposes. Wrong shape → wrong topology → wrong gradient → penalty for that shape next tick. Right shape → reinforced.

### 5.4 Initialisation

All $\mathbf{u}_{i,j} = \mathbf{0}$ except waypoints, which are clamped immediately. The first $T_{\text{warm}}$ ticks run diffusion only (no Layer-1 update), giving the field a chance to spread before shapes begin reacting to it.

---

## 6. The Coupling — Reward Score

For cell $(i,j)$, candidate shape $s$ with open ports $\{a, b\}$ (or $\{a\}$ for endpoints), define:

### 6.1 Per-segment alignment

For each segment $k$ and each open port direction $d$, the **outward neighbour** is the cell across that port. Its concentration of chemical $k$ is denoted $u_d^k$. The cell's own concentration is $u_0^k$. The directional gradient at port $d$ for chemical $k$ is:

$$g_d^k = u_d^k - u_0^k$$

Positive $g_d^k$ means moving through port $d$ goes *up* the gradient of chemical $k$.

### 6.2 Best-segment-pair score

For a through-shape with open ports $\{a, b\}$, the score is the maximum, over consecutive segment pairs $(k, k+1)$, of:

$$\sigma(s) = \max_{k} \left[ g_a^k + g_b^{k+1} \right] \text{ or symmetrically } \left[ g_a^{k+1} + g_b^k \right]$$

i.e. the cell sits along the channel for *some* segment $k$ if one of its open ports flows up segment-$k$'s gradient and the other up segment-$(k+1)$'s. The cell finds the best such segment pair and uses that score.

For an endpoint shape (cell is a waypoint with one open port $a$), the score is simply $g_a^k$ for the single adjacent segment $k$ — the path leaves the endpoint along the channel toward the next waypoint.

### 6.3 Thresholds

$\theta_+ > 0$ and $\theta_- < 0$ are small constants (e.g. $\pm 10^{-3}$) defining a dead band where the cell is uncertain and updates nothing. This dampens oscillations.

---

## 7. Controlled Stochastic Noise

To escape symmetric local minima (e.g. two near-identical candidate paths producing tied gradients), each cell at each tick performs:

1. With probability $\eta(t)$, replace $P_{i,j}$ with a soft mixture:
   $$P_{i,j} \leftarrow (1 - \beta) \cdot P_{i,j} + \beta \cdot \text{Uniform}(\mathcal{A}_{i,j})$$
   for small $\beta \in (0, 0.5)$.
2. The decision is made by the cell's own local PRNG seeded from `(i, j, run_id, tick)` — no global RNG, so the noise is itself a local CA rule.

$\eta(t)$ may be annealed: high early to encourage exploration, low late to allow convergence. A reasonable schedule is $\eta(t) = \eta_0 \cdot \exp(-t / \tau)$.

---

## 8. Joint Update Algorithm

```
Algorithm 1 — Joint dual-layer CA tick
─────────────────────────────────────
Inputs : grid state {P, s, u} at tick t
Output : grid state at tick t+1

# Step 1: Diffuse chemicals through current path topology
for each segment chemical k:
    for each cell (i, j) in parallel:
        Ω ← neighbours (p,q) such that edge(i,j ↔ p,q) is open
                                  in BOTH s[i,j] and s[p,q]
                          AND not walled
        if Ω is empty:
            mix ← u[i,j,k]
        else:
            mix ← (1-α) · u[i,j,k] + α · mean( u[p,q,k] for (p,q) in Ω )
        u_new[i,j,k] ← (1-δ) · mix

# Step 2: Re-assert waypoint sources
for each waypoint W_k:
    if k > 1:  u_new[W_k, k-1] ← C₀
    if k < K:  u_new[W_k, k]   ← C₀

u ← u_new

# Step 3: Score and update shape probabilities
for each cell (i, j) in parallel:
    for each shape s' in A[i,j]:
        σ ← score(s', neighbours of (i,j), u)   # §6
        if σ > θ_+:  P[i,j,s'] ← P[i,j,s'] · r
        elif σ < θ_-: P[i,j,s'] ← P[i,j,s'] · p
    P[i,j] ← normalize(P[i,j])

# Step 4: Optional per-cell stochastic perturbation
for each cell (i, j) in parallel:
    if rng(i, j, run_id, t) < η(t):
        P[i,j] ← (1-β) · P[i,j] + β · uniform(A[i,j])

# Step 5: Re-select chosen shapes
for each cell (i, j) in parallel:
    s[i,j] ← argmax P[i,j]

return {P, s, u}
```

All four steps are local: every cell reads only its own state and that of its 4-neighbours.

---

## 9. Termination, Restart, and Validation

### 9.1 Convergence detection

The CA is *quiescent* at tick $t$ when no cell has changed its argmax shape for $T_{\text{stable}}$ consecutive ticks. Quiescence is necessary but not sufficient — the system might be stuck.

### 9.2 Path validation

When quiescent, run a **trace from waypoint #1**: follow the open ports of the chosen shapes step-by-step. The trace succeeds iff:

1. It visits every cell exactly once.
2. It encounters waypoints in the correct order $1, 2, \dots, K$.
3. It terminates at waypoint $K$.

A successful trace → **solution found, halt**.

### 9.3 Timeout and restart

If the CA has not produced a valid trace after $T_{\max}$ ticks, the run is abandoned. A new run begins with:

- Fresh `run_id` (re-seeds the per-cell noise PRNG).
- Re-initialised $P_{i,j}$ — uniform over $\mathcal{A}_{i,j}$ (or perturbed-uniform).
- Re-zeroed $\mathbf{u}_{i,j}$ except waypoint clamps.

The solver retries up to $R_{\max}$ times before reporting failure.

### 9.4 Failure reporting

If $R_{\max}$ runs all time out, the solver returns:
- The best partial trace seen across runs (longest valid prefix from #1).
- Per-run quiescence statistics for diagnostics.
- A flag indicating "no solution found within budget."

---

## 10. Visualisation

Two synchronised panels rendered every $V$ ticks:

**Panel A — Path Layer.** The grid drawn as cells; each cell's chosen shape rendered as line segments connecting the appropriate edge midpoints. Cell background tinted by the argmax probability (white = uncertain, dark = confident). Walls drawn as thick borders. Waypoint numbers overlaid.

**Panel B — Chemical Layer.** Either:
- **Multi-channel mode:** $K-1$ small heatmaps in a strip, one per segment. Each cell's brightness = its concentration of that segment's chemical.
- **Composite mode:** single heatmap; per-cell hue = $\arg\max_k u[k]$ (which segment it most belongs to); value = $\max_k u[k]$.

Composite is the prettier default; multi-channel is the diagnostic view.

Output formats: live `matplotlib.animation.FuncAnimation`, MP4 export, and PNG snapshots at quiescence.

---

## 11. Data Model

This section pins the data structures the implementation must use. Decisions here are deliberately conservative: the goal is to make whole classes of bugs unrepresentable rather than to maximise expressiveness.

### 11.1 Puzzle artifact

The puzzle is the immutable input specification. It lives on disk as JSON and in memory as a frozen dataclass.

#### 11.1.1 Coordinate convention (binding)

| Convention | Value |
|---|---|
| Origin `(0, 0)` | top-left cell |
| Row index `i` | increases downward |
| Column index `j` | increases rightward |
| Direction `N` | `(i−1, j)`, i.e. one row up |
| Direction `E` | `(i, j+1)` |
| Direction `S` | `(i+1, j)` |
| Direction `W` | `(i, j−1)` |

A `Direction` enum is the only sanctioned way to refer to ports anywhere in the codebase. Strings `"N"`/`"E"`/`"S"`/`"W"` appear only at the JSON parser boundary.

#### 11.1.2 JSON wire format (sparse)

```json
{
  "size": 6,
  "waypoints": [
    {"row": 0, "col": 0, "number": 1},
    {"row": 2, "col": 2, "number": 2},
    {"row": 5, "col": 5, "number": 6}
  ],
  "walls": [
    {"row": 2, "col": 3, "blocked": ["N", "E"]}
  ],
  "_comment": "Optional human-readable annotation, ignored by parser",
  "name": "Optional puzzle name",
  "source": "Optional provenance (e.g. 'LinkedIn 2026-04-15')"
}
```

The parser **must reject** any unknown top-level or nested key (typo guard).

#### 11.1.3 In-memory representation

```python
@dataclass(frozen=True, slots=True)
class Waypoint:
    row: int
    col: int
    number: int

@dataclass(frozen=True, slots=True)
class Edge:
    """Canonical undirected edge between two cells (a, b) with a < b lexicographically."""
    a: tuple[int, int]
    b: tuple[int, int]

@dataclass(frozen=True, slots=True)
class Puzzle:
    size: int
    waypoints: tuple[Waypoint, ...]      # sorted by .number
    walled_edges: frozenset[Edge]        # canonical, no redundancy possible
    name: str | None = None
    source: str | None = None
```

The JSON wire format uses *per-cell* wall lists for human convenience, but the in-memory `Puzzle` stores walls as a `frozenset[Edge]` of canonical edges. Conversion happens once, at load time.

#### 11.1.4 Validation invariants (enforced on load)

The parser raises `PuzzleValidationError` if any of these fail:

1. `size ≥ 2`.
2. Every waypoint coordinate satisfies `0 ≤ row, col < size`.
3. Waypoint numbers form exactly `{1, 2, …, K}` for some `K ≥ 2` — no gaps, no duplicates.
4. No two waypoints share a coordinate.
5. Every walled-cell coordinate is in bounds.
6. Every wall direction string is in `{"N", "E", "S", "W"}`.
7. Wall consistency: if cell `A` blocks direction `D` and the cell on the other side of edge `D` exists, then either that cell does not list the opposing direction, or it does and the two agree. **Disagreement is an error**, not a silent override.
8. No unknown keys in the JSON object (strict schema).

After validation, walls are canonicalised: every walled edge is recorded exactly once in `walled_edges`, regardless of how many times it appeared in the per-cell JSON listing.

### 11.2 Engine state

The engine state is the mutable working memory of one solver run. Constructed fresh per run; never reset in place.

#### 11.2.1 Structure

```python
@dataclass(slots=True)
class EngineState:
    puzzle: Puzzle                          # immutable backreference
    run_id: int                             # seeds the per-cell PRNG
    tick: int                               # monotonically increasing
    probs:   NDArray[np.float64]            # (N, N, |S|) shape probabilities
    shapes:  NDArray[np.int8]               # (N, N) argmax shape index
    chems:   NDArray[np.float32]            # (N, N, K-1) segment concentrations
    allowed: NDArray[np.bool_]              # (N, N, |S|) action mask, fixed for puzzle lifetime
```

`|S| = 10` (six through-shapes + four endpoint-shapes). All arrays are C-contiguous.

#### 11.2.2 Construction (the only entry point)

```python
@classmethod
def fresh(cls, puzzle: Puzzle, run_id: int) -> "EngineState": ...
```

`fresh()` is the **only way** to obtain an `EngineState`. It allocates new arrays, computes the `allowed` mask from the puzzle's walls and waypoint constraints, initialises `probs` to uniform-over-allowed, zeros `chems` (sources are re-asserted on the first diffusion step), and sets `tick = 0`.

There is no `reset()` method. To restart, the runner discards the current state object and calls `fresh()` again with a new `run_id`.

#### 11.2.3 Tick semantics

```python
def tick(self) -> None: ...
```

Mutates `self.probs`, `self.shapes`, `self.chems`, and `self.tick` in place. Returns `None` to signal side effects. Helpers invoked from `tick()` operate on `self`'s arrays directly; no helper accepts an external mutable array.

#### 11.2.4 Snapshots

```python
def snapshot(self) -> "EngineStateSnapshot": ...
```

Returns a `frozen` dataclass holding deep copies of all arrays plus the scalar fields. Snapshots are the **only object visualisation and tests are allowed to consume**. Function signatures in `visualization.py` and `tests/` accept `EngineStateSnapshot`, never `EngineState`. Misuse becomes a static type error under `pyright`.

#### 11.2.5 Aliasing discipline

Internal helpers receive arrays as views. To prevent accidental mutation through helper functions, tests run with `array.flags.writeable = False` set on snapshot arrays after copy. Production code does not need to enforce this — only `EngineState.tick` is permitted to write to `self.*` arrays, and code review enforces this rule.

### 11.3 Reproducibility invariant

Two runs with identical `(puzzle, run_id)` must produce byte-equal snapshot sequences.

#### 11.3.1 PRNG seeding

The per-cell stochastic noise (§7) draws from `numpy.random.Generator` seeded as:

```python
seed = hash((i, j, run_id, tick)) & 0xFFFF_FFFF_FFFF_FFFF
rng = np.random.default_rng(seed)
```

No global RNG is read from anywhere in the tick path. Cell `(i, j)` at tick `t` always produces the same noise decision for fixed `run_id`.

#### 11.3.2 Test enforcement

A test in `tests/test_reproducibility.py` MUST exist, asserting:

```python
def test_byte_equal_under_repeat():
    puzzle = load_puzzle("puzzles/tiny.json")
    state_a = EngineState.fresh(puzzle, run_id=42)
    state_b = EngineState.fresh(puzzle, run_id=42)
    for _ in range(100):
        state_a.tick()
        state_b.tick()
    snap_a = state_a.snapshot()
    snap_b = state_b.snapshot()
    assert np.array_equal(snap_a.probs, snap_b.probs)
    assert np.array_equal(snap_a.shapes, snap_b.shapes)
    assert np.array_equal(snap_a.chems, snap_b.chems)
```

This test is a hard correctness gate. Any code change that breaks it must either be reverted or accompanied by a documented justification.

### 11.4 Test puzzle layout

```
puzzles/
├── tiny_3x3.json              # input only — what the solver sees
├── tiny_3x3.solution.json     # expected path — never read by solver code
├── linkedin_2026_04_15.json
├── linkedin_2026_04_15.solution.json
└── ...
```

#### 11.4.1 Solution file format

```json
{
  "size": 3,
  "path": [[0, 0], [0, 1], [0, 2], [1, 2], [1, 1], [1, 0], [2, 0], [2, 1], [2, 2]]
}
```

The `path` is the full Hamiltonian sequence as `(row, col)` cells. A test helper compares a solver's emitted path against this list.

#### 11.4.2 Read isolation

Solver code (anything under `src/zip_ca/` excluding `cli.py` test fixtures) MUST NOT import any module that reads `*.solution.json`. A linting rule (manual code review for now; Ruff custom check later) enforces this.

---

## 12. Hyperparameters (initial guesses)

| Symbol | Meaning | Initial value | Notes |
|---|---|---|---|
| $C_0$ | Source emission | $1.0$ | Sets scale; gradients are scale-equivariant. |
| $\alpha$ | Diffusion rate | $0.25$ | Stable for 5-point stencil up to $0.25$. |
| $\delta$ | Decay per tick | $0.01$ | Prevents drift far from sources. |
| $r$ | Reward factor | $1.05$ | Mild — paper uses similar magnitudes. |
| $p$ | Penalty factor | $0.95$ | $r \cdot p \approx 1$ keeps probabilities well-behaved. |
| $\theta_\pm$ | Score dead band | $\pm 10^{-3}$ | Tune by inspection of score histograms. |
| $\eta_0$ | Initial noise prob | $0.10$ | Annealed. |
| $\tau$ | Noise time constant | $200$ ticks | Decays roughly over a run. |
| $\beta$ | Noise mixing | $0.20$ | Soft pull toward uniform when triggered. |
| $T_{\text{warm}}$ | Warm-up ticks | $20$ | Diffusion only, no Layer-1 updates. |
| $T_{\text{stable}}$ | Quiescence window | $30$ | No argmax flips for this many ticks. |
| $T_{\max}$ | Per-run cap | $5{,}000$ | Plenty for $\le 8\times 8$. |
| $R_{\max}$ | Restart cap | $20$ | Solver-level. |

All to be tuned empirically once Phase 5 lands.

---

## 12. Implementation Roadmap

In strict order; later phases assume earlier ones are verified.

1. **Puzzle data model + parser.** JSON schema for puzzles (`{size, waypoints, walls}`), parser, validator. Round-trip serialisation tests.
2. **Path-shape engine alone.** `Shape` enum, port logic, $\mathcal{A}_{i,j}$ computation honouring walls + waypoints. Render Panel A with hand-set shapes; visually verify.
3. **Diffusion layer alone.** With shapes *fixed* to a known correct path, run pure diffusion, render Panel B, eyeball the gradient field. No Layer-1 update yet.
4. **Coupling.** Wire score → reward → re-select. Run on a $3 \times 3$ trivial puzzle (e.g. waypoint 1 at one corner, 2 at opposite corner). Solver must converge to *the* unique path.
5. **Real puzzles + noise + restart.** Hand-type 5–10 LinkedIn-grade puzzles. Add stochastic noise and the run-restart loop. Aim for ≥80% solve rate within $R_{\max}$.
6. **Visualisation polish.** Animations, MP4 export, side-by-side composite view.

Testing exists from Phase 1 onward via `pytest`. Per project convention, tests are written when explicitly requested.

---

## 13. Open Questions / Future Work

- **Selective probability reset (paper §5).** Would porting this enhancement help recover from disconnected-island lock-in faster than restarting?
- **Variable learning rates (paper §5).** Stronger reward/penalty for cells with high score magnitude — worth exploring after baseline.
- **Adaptive noise schedule.** Could $\eta(t)$ be driven by per-cell quiescence detectors rather than a global timer? More CA-faithful.
- **Solution uniqueness.** Standard Zip puzzles have unique solutions. Does this CA's emergent dynamic ever produce alternative valid paths in puzzles where they exist? Useful diagnostic.
- **Empirical phase diagram.** Sweep $(\alpha, \delta, r, p)$ — at what point does the system lose convergence ability?

---

## 14. References

- Chatzinikolaou, T. P., Karamani, R.-E., Fyrigos, I.-A., & Sirakoulis, G. Ch. (2024). Handling Sudoku puzzles with irregular learning cellular automata. *Natural Computing*, 23(1), 41–60. https://doi.org/10.1007/s11047-024-09975-4
- Tero, A., Takagi, S., Saigusa, T., Ito, K., Bebber, D. P., Fricker, M. D., Yumiki, K., Kobayashi, R., & Nakagaki, T. (2010). Rules for biologically inspired adaptive network design. *Science*, 327(5964), 439–442.
- Nakagaki, T., Yamada, H., & Tóth, Á. (2000). Maze-solving by an amoeboid organism. *Nature*, 407, 470.
- Narendra, K. S., & Thathachar, M. A. L. (1974). Learning automata — A survey. *IEEE Transactions on Systems, Man, and Cybernetics*, SMC-4(4), 323–334.
