# Epidemic Simulation Models

All models are implemented on a weighted temporal network in which each snapshot corresponds to one month of sexual contact data. Edge weights represent the number of contacts between two nodes within the same month. Time progresses one snapshot at a time. Each simulation starts with **10 randomly chosen infected nodes**.

---

## Behaviour Assignment

Before each simulation, every node is assigned a **transmission rate** using `assign_transmission_rates` (real-data network) or `assign_transmission_rates_generic` (synthetic network). Two transmission rates are used:

| Parameter | Value |
|-----------|-------|
| β_high (high-risk behaviour) | 0.05 |
| β_low  (low-risk behaviour)  | 0.001 |

70 % of nodes are assigned β_high; the remaining 30 % are assigned β_low. Three criteria determine which nodes receive the high rate:

- **random** – each node independently receives β_high with probability 0.70.
- **high_to_hub** – the top 70 % of nodes by aggregate degree (number of unique partners across all snapshots) receive β_high.
- **high_to_low** – the bottom 30 % of nodes by aggregate degree receive β_high.

The threshold between hub and non-hub is set at the 30th percentile of the aggregate degree distribution (`np.quantile(degrees, 0.30)`).

---

## Risk Categories

Nodes are stratified into four risk categories by crossing two binary dimensions:

- **Contact activity**: *high contact* if a node's aggregate degree ≥ mean aggregate degree; *low contact* otherwise.
- **Behaviour**: *high risk* if the node's assigned transmission rate equals β_high (0.05); *low risk* if it equals β_low (0.001).

| Category | Contact | Behaviour |
|----------|---------|-----------|
| High contact / High risk | ≥ mean degree | β_high |
| Low contact  / High risk | < mean degree | β_high |
| High contact / Low risk  | ≥ mean degree | β_low  |
| Low contact  / Low risk  | < mean degree | β_low  |

---

## SI Model

### Compartments

| Compartment | Description |
|-------------|-------------|
| **S** | Susceptible – initial state of all nodes except the seeds |
| **I** | Infected – transmits the pathogen to susceptible neighbours |

### Transition rules

**S → I**: At each snapshot *t*, for every edge *(u, v, w)* connecting an infected node *u* to a susceptible node *v*, the susceptible node becomes infected with probability

$$p = \min(1,\; w \times \beta_u)$$

where *w* is the edge weight (number of contacts in that snapshot) and β_u is the transmission rate assigned to *u*. The model is absorbing: once infected, a node remains infected for the rest of the simulation.

### Parameters

| Parameter | Value |
|-----------|-------|
| Initial infected (seeds) | 10, chosen uniformly at random |
| β_high | 0.05 |
| β_low  | 0.001 |
| Fraction of nodes with β_high | 70 % |
| Number of simulations per criterion | 100 (synthetic) / 1000 (real data) |

---

## SIS Model

### Compartments

| Compartment | Description |
|-------------|-------------|
| **S** | Susceptible – can be re-infected after recovery |
| **I** | Infected – transmits the pathogen; recovers after a fixed time |

### Transition rules

**S → I**: Identical to the SI model — probability min(1, w × β_u) upon contact with an infected node.

**I → S (recovery)**: An infected node recovers and returns to the susceptible state after **1 year** (12 snapshots). On recovery, no immunity is acquired; the node is immediately susceptible again.

### Parameters

| Parameter | Value |
|-----------|-------|
| Initial infected (seeds) | 10, chosen uniformly at random |
| β_high | 0.05 |
| β_low  | 0.001 |
| Fraction of nodes with β_high | 70 % |
| Recovery time | 1 year = 12 snapshots |
| Number of simulations per criterion | 100 (synthetic) / 1000 (real data) |

---

## SITS Model

### Compartments

| Compartment | Description |
|-------------|-------------|
| **S** | Susceptible – initial state |
| **I** | Infected, **untested** – keeps its originally assigned transmission rate (β_high or β_low) |
| **T** | Infected, **tested** – transmission rate reduced to β_tested after diagnosis |

### Transition rules

**S → I**: At each snapshot, for every edge *(u, v, w)* with *u* infected (I or T) and *v* susceptible, *v* becomes infected with probability

$$p = \min(1,\; w \times \beta_{\text{current},u})$$

where β_current is β_high or β_low for an untested node (I) and β_tested for a tested node (T).

**I → T (testing)**: In the snapshot **immediately after** a node enters compartment I, it is tested with probability *p* (default 0.5). If tested, the node moves to T and its transmission rate is reduced to β_tested = 0.001.
<!--  -->
**I → T does not happen**: With probability 1 − *p*, the node remains in I and keeps its originally assigned transmission rate.

**I → S (recovery without testing)**: An untested infected node recovers and returns to S after **1 year** (12 snapshots). Its originally assigned transmission rate (β_high or β_low) is restored upon recovery.

**T → S (recovery after testing)**: A tested node recovers and returns to S after **5 months** (5 snapshots). Its originally assigned transmission rate is restored upon recovery.

### Parameters

| Parameter | Value |
|-----------|-------|
| Initial infected (seeds) | 10, chosen uniformly at random |
| β_high | 0.05 |
| β_low  | 0.001 |
| β_tested | 0.001 |
| Fraction of nodes with β_high | 70 % |
| Testing probability *p* | 0.5 (default) |
| Recovery time for I | 1 year = 12 snapshots |
| Recovery time for T | 5 months = 5 snapshots |
| Number of simulations per criterion | 100 |

### Outbreak size vs. testing probability

To assess the effect of the testing intervention, the SITS model is also run sweeping *p* over 10 equally spaced values from 0 to 1 (inclusive: 0.00, 0.11, 0.22, …, 1.00). For each value of *p*, `N_SIMULATIONS` independent simulations are run and summarised with `summarize_epidemic` (see [Outcome measures](#outcome-measures)). Two plots are produced: a box plot of the **total number of nodes ever infected, restricted to outbreak runs**, and the **outbreak probability** as a function of *p*.

---

## SITp Model (Periodic Testing)

### Compartments

| Compartment | Description |
|-------------|-------------|
| **S** | Susceptible – initial state |
| **I** | Infected, **untested** – keeps its originally assigned transmission rate |
| **T** | Infected, **tested** – transmission rate reduced to β_tested after diagnosis |

### Transition rules

**S → I**: Same as all other models — probability min(1, w × β_current,u) upon contact with an infected node.

**I → T (periodic testing)**: Each node's first scheduled test occurs at a uniformly random offset in [0, T_t) snapshots after the simulation start; thereafter, testing occurs every T_t snapshots. At each scheduled test:
- S nodes: unaffected.
- I nodes: move to T; β drops to β_tested.
- T nodes: recovery clock resets (re-tested; 5 more snapshots until recovery).

**T → S (recovery after testing)**: A tested node returns to S after **5 months** (5 snapshots). Its originally assigned transmission rate is restored upon recovery.

**I nodes never recover on their own** (no spontaneous recovery from the untested infected state).

### Parameters

| Parameter | Value |
|-----------|-------|
| Initial infected (seeds) | 10, chosen uniformly at random |
| β_high | 0.05 |
| β_low  | 0.001 |
| β_tested | 0.001 |
| Fraction of nodes with β_high | 70 % |
| Testing interval T_t | 6, 12, or 24 months (epidemic plot); swept 6–24 months (boxplot) |
| Initial test offset | uniform in [0, T_t) per node |
| Recovery time for T | 5 months = 5 snapshots |
| Number of simulations | 100 (synthetic) / 1000 (real data) |

### Outbreak size vs. testing interval

The SITp model is also run sweeping T_t over 15 equally spaced values from 6 to 24 months (rounded to integers). For each T_t, `N_SIMULATIONS` independent simulations are run and summarised with `summarize_epidemic`. As for SITS, two plots are produced: the **total number of nodes ever infected in outbreak runs** (box plot, x-axis in months) and the **outbreak probability** vs. T_t.

---

<a id="outcome-measures"></a>
## Outcome measures

The peak number of infected is noisy, and when the epidemic does not take off it just equals the number of seeds. The SITS and SITp sweeps are therefore summarised with `summarize_epidemic(counts, outbreak_threshold=50, last_n=12)`, which returns for each run:

| Output | Definition |
|--------|------------|
| `ever` | Cumulative incidence: number of distinct nodes ever infected (I or T), seeds included |
| `prevalence` | Mean number of infected (I + T) over the last `last_n` = 12 snapshots |
| `outbreak` | True if `ever` > `outbreak_threshold` (default 50) |
| `p_outbreak` | Fraction of runs that are outbreaks |

The size box plots only show outbreak runs, so they are not dominated by runs that die out immediately. The probability of an outbreak is shown separately.
