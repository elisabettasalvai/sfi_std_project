import numpy as np
from collections import Counter


def assign_transmission_rates(df_contacts, criterion='random',
                               beta_low=0.001, beta_high=0.05,
                               high_degree_fraction=0.70):
    """
    Assign a transmission rate to every node in the dataset.

    Parameters
    ----------
    df_contacts : DataFrame
        Full contact dataframe with Female_id, Male_id columns.
    criterion : str
        'random'      — high/low assigned at random
        'high_to_hub' — high beta assigned to high-degree nodes (top fraction)
        'high_to_low' — high beta assigned to low-degree nodes (bottom fraction)
    beta_low : float
        Transmission rate for low-beta nodes.
    beta_high : float
        Transmission rate for high-beta nodes.
    high_degree_fraction : float
        Fraction of nodes considered high-degree (default 0.70 → top 70%).

    Returns
    -------
    dict : node_id -> beta
    """
    agg_degree = Counter()
    agg_degree.update(df_contacts['Female_id'])
    agg_degree.update(df_contacts['Male_id'])

    all_nodes = np.array(list(agg_degree.keys()))
    degrees   = np.array([agg_degree[n] for n in all_nodes])

    threshold = np.quantile(degrees, 1.0 - high_degree_fraction)
    is_hub    = degrees >= threshold

    if criterion == 'random':
        high_mask = np.random.rand(len(all_nodes)) < high_degree_fraction
    elif criterion == 'high_to_hub':
        high_mask = is_hub
    elif criterion == 'high_to_low':
        high_mask = ~is_hub
    else:
        raise ValueError(
            f"criterion must be 'random', 'high_to_hub', or 'high_to_low'. Got '{criterion}'."
        )

    return {
        node: (beta_high if h else beta_low)
        for node, h in zip(all_nodes, high_mask)
    }


def _build_weighted_edges(df_contacts):
    """Return a dict month -> list of (Female_id, Male_id, weight)."""
    df_w = (
        df_contacts
        .groupby(['Female_id', 'Male_id', 'Month_count'])
        .size()
        .reset_index(name='weight')
    )
    return {
        month: list(zip(grp['Female_id'], grp['Male_id'], grp['weight']))
        for month, grp in df_w.groupby('Month_count')
    }


def run_SI_simulations(df_contacts, beta_map,
                       n_simulations=100, n_seeds=10):
    """
    Run multiple SI epidemic simulations on the weighted temporal network.

    Transmission probability for an edge (u→v) in snapshot t:
        p = min(1, weight(u,v,t) * beta(u))

    Parameters
    ----------
    df_contacts : DataFrame
        Full contact dataframe with Female_id, Male_id, Month_count.
    beta_map : dict
        node_id -> base transmission rate (from assign_transmission_rates).
    n_simulations : int
        Number of independent simulations to run.
    n_seeds : int
        Number of randomly chosen infected nodes at snapshot 1.

    Returns
    -------
    infected_counts : np.ndarray, shape (n_simulations, n_snapshots)
    snapshots : list
    """
    snapshots  = sorted(df_contacts['Month_count'].unique())
    all_nodes  = list(set(df_contacts['Female_id']) | set(df_contacts['Male_id']))
    edges_by_month = _build_weighted_edges(df_contacts)

    infected_counts = np.zeros((n_simulations, len(snapshots)), dtype=int)

    for sim in range(n_simulations):
        infected = set(np.random.choice(all_nodes, size=n_seeds, replace=False))

        for t, month in enumerate(snapshots):
            new_infected = set()
            for u, v, w in edges_by_month.get(month, []):
                if u in infected and v not in infected:
                    if np.random.rand() < min(1.0, w * beta_map.get(u, 0.001)):
                        new_infected.add(v)
                if v in infected and u not in infected:
                    if np.random.rand() < min(1.0, w * beta_map.get(v, 0.001)):
                        new_infected.add(u)
            infected |= new_infected
            infected_counts[sim, t] = len(infected)

    return infected_counts, snapshots


def run_SIS_simulations(df_contacts, beta_map,
                        n_simulations=100, n_seeds=10,
                        recovery_time_years=1.0):
    """
    Run multiple SIS epidemic simulations on the weighted temporal network.

    Transmission (S→I): probability = min(1, weight * beta(u)).
    Recovery   (I→S):   deterministic after round(recovery_time_years * 12) snapshots.

    Parameters
    ----------
    df_contacts : DataFrame
        Full contact dataframe with Female_id, Male_id, Month_count.
    beta_map : dict
        node_id -> base transmission rate (from assign_transmission_rates).
    n_simulations : int
        Number of independent simulations to run.
    n_seeds : int
        Number of randomly chosen infected nodes at snapshot 1.
    recovery_time_years : float
        Time in years after which an infected node recovers (default 1 year = 12 snapshots).

    Returns
    -------
    infected_counts : np.ndarray, shape (n_simulations, n_snapshots)
    snapshots : list
    """
    snapshots  = sorted(df_contacts['Month_count'].unique())
    all_nodes  = list(set(df_contacts['Female_id']) | set(df_contacts['Male_id']))
    edges_by_month = _build_weighted_edges(df_contacts)
    recovery_snapshots = round(recovery_time_years * 12)

    infected_counts = np.zeros((n_simulations, len(snapshots)), dtype=int)

    for sim in range(n_simulations):
        infection_time = {
            node: 0
            for node in np.random.choice(all_nodes, size=n_seeds, replace=False)
        }

        for t, month in enumerate(snapshots):
            recovered = {
                node for node, t_inf in infection_time.items()
                if (t - t_inf) >= recovery_snapshots
            }
            for node in recovered:
                del infection_time[node]

            infected = set(infection_time.keys())

            new_infected = {}
            for u, v, w in edges_by_month.get(month, []):
                if u in infected and v not in infected and v not in new_infected:
                    if np.random.rand() < min(1.0, w * beta_map.get(u, 0.001)):
                        new_infected[v] = t
                if v in infected and u not in infected and u not in new_infected:
                    if np.random.rand() < min(1.0, w * beta_map.get(v, 0.001)):
                        new_infected[u] = t

            infection_time.update(new_infected)
            infected_counts[sim, t] = len(infection_time)

    return infected_counts, snapshots


# ---------------------------------------------------------------------------
# Generic variants — work with any node/edge format
# ---------------------------------------------------------------------------

def assign_transmission_rates_generic(degree_map, criterion='random',
                                       beta_low=0.001, beta_high=0.05,
                                       high_degree_fraction=0.70):
    """
    Assign a transmission rate to every node given a pre-computed degree map.

    Parameters
    ----------
    degree_map : dict  node_id -> aggregate degree across all snapshots
    criterion  : 'random' | 'high_to_hub' | 'high_to_low'
    """
    all_nodes = np.array(list(degree_map.keys()))
    degrees   = np.array([degree_map[n] for n in all_nodes])
    threshold = np.quantile(degrees, 1.0 - high_degree_fraction)
    is_hub    = degrees >= threshold

    if criterion == 'random':
        high_mask = np.random.rand(len(all_nodes)) < high_degree_fraction
    elif criterion == 'high_to_hub':
        high_mask = is_hub
    elif criterion == 'high_to_low':
        high_mask = ~is_hub
    else:
        raise ValueError(
            f"criterion must be 'random', 'high_to_hub', or 'high_to_low'. Got '{criterion}'."
        )
    return {node: (beta_high if h else beta_low) for node, h in zip(all_nodes, high_mask)}


def run_SI_simulations_generic(edges_by_snapshot, all_nodes, beta_map,
                                n_simulations=100, n_seeds=10):
    """
    SI simulation for a generic weighted temporal network.

    Parameters
    ----------
    edges_by_snapshot : dict  snapshot_key -> list of (u, v, weight)
    all_nodes         : list  all node ids
    beta_map          : dict  node_id -> transmission rate
    """
    snapshots = sorted(edges_by_snapshot.keys())
    infected_counts = np.zeros((n_simulations, len(snapshots)), dtype=int)

    for sim in range(n_simulations):
        infected = set(np.random.choice(all_nodes, size=n_seeds, replace=False))
        for t, snap in enumerate(snapshots):
            new_infected = set()
            for u, v, w in edges_by_snapshot[snap]:
                if u in infected and v not in infected:
                    if np.random.rand() < min(1.0, w * beta_map.get(u, 0.001)):
                        new_infected.add(v)
                if v in infected and u not in infected:
                    if np.random.rand() < min(1.0, w * beta_map.get(v, 0.001)):
                        new_infected.add(u)
            infected |= new_infected
            infected_counts[sim, t] = len(infected)

    return infected_counts, snapshots


def run_SIS_simulations_generic(edges_by_snapshot, all_nodes, beta_map,
                                 n_simulations=100, n_seeds=10,
                                 recovery_time_years=1.0):
    """
    SIS simulation for a generic weighted temporal network.

    Parameters
    ----------
    edges_by_snapshot : dict  snapshot_key -> list of (u, v, weight)
    all_nodes         : list  all node ids
    beta_map          : dict  node_id -> transmission rate
    recovery_time_years : float  each snapshot = 1 month
    """
    snapshots = sorted(edges_by_snapshot.keys())
    recovery_snapshots = round(recovery_time_years * 12)
    infected_counts = np.zeros((n_simulations, len(snapshots)), dtype=int)

    for sim in range(n_simulations):
        infection_time = {
            node: 0
            for node in np.random.choice(all_nodes, size=n_seeds, replace=False)
        }

        for t, snap in enumerate(snapshots):
            recovered = {node for node, t_inf in infection_time.items()
                         if (t - t_inf) >= recovery_snapshots}
            for node in recovered:
                del infection_time[node]
            infected = set(infection_time.keys())
            new_infected = {}
            for u, v, w in edges_by_snapshot[snap]:
                if u in infected and v not in infected and v not in new_infected:
                    if np.random.rand() < min(1.0, w * beta_map.get(u, 0.001)):
                        new_infected[v] = t
                if v in infected and u not in infected and u not in new_infected:
                    if np.random.rand() < min(1.0, w * beta_map.get(v, 0.001)):
                        new_infected[u] = t
            infection_time.update(new_infected)
            infected_counts[sim, t] = len(infection_time)

    return infected_counts, snapshots


# ---------------------------------------------------------------------------
# SITS model
# ---------------------------------------------------------------------------

def _run_SITS_core(edges_by_snapshot, all_nodes, beta_map,
                   n_simulations, n_seeds, p_test,
                   recovery_I_snaps, recovery_T_snaps, beta_tested):
    """
    Shared SITS simulation logic (called by the real-data and generic wrappers).

    Compartments
    ------------
    S  susceptible (initial)
    I  infected, untested  — keeps original beta_map transmission rate
    T  infected, tested    — transmission rate reduced to beta_tested

    Transitions
    -----------
    S -> I : contact with any infected (I or T) node in snapshot t;
             prob = min(1, weight * beta_current[infector])
    I -> T : with probability p_test in the snapshot immediately after infection
    I -> S : after recovery_I_snaps snapshots; original beta restored
    T -> S : after recovery_T_snaps snapshots; original beta restored

    Returns
    -------
    counts    : dict with keys 'S', 'I', 'T', 'ever', each ndarray (n_simulations, n_snapshots)
                'ever' counts cumulative unique nodes that have entered I or T at any point
    snapshots : list of ordered snapshot keys
    """
    snapshots = sorted(edges_by_snapshot.keys())
    n_snaps   = len(snapshots)

    counts_S    = np.zeros((n_simulations, n_snaps), dtype=int)
    counts_I    = np.zeros((n_simulations, n_snaps), dtype=int)
    counts_T    = np.zeros((n_simulations, n_snaps), dtype=int)
    counts_ever = np.zeros((n_simulations, n_snaps), dtype=int)

    for sim in range(n_simulations):
        time_I        = {}   # node -> t when entered I
        time_T        = {}   # node -> t when entered T
        pending_test  = {}   # nodes newly infected this snapshot; testing resolved next snapshot
        beta_current  = dict(beta_map)
        ever_infected = set()

        # Seed infections: all seeds start as I, with testing resolved at t=0
        for node in np.random.choice(all_nodes, size=n_seeds, replace=False):
            time_I[node]       = 0
            pending_test[node] = 0
            ever_infected.add(node)

        for t, snap in enumerate(snapshots):

            # 1. Recovery
            for node in [n for n, t0 in list(time_I.items()) if (t - t0) >= recovery_I_snaps]:
                del time_I[node]
                beta_current[node] = beta_map[node]
            for node in [n for n, t0 in list(time_T.items()) if (t - t0) >= recovery_T_snaps]:
                del time_T[node]
                beta_current[node] = beta_map[node]

            # 2. Testing decision for nodes infected in the previous snapshot
            for node in list(pending_test):
                if node in time_I:                       # still infected and untested
                    if np.random.rand() < p_test:
                        time_T[node] = t
                        del time_I[node]
                        beta_current[node] = beta_tested
            pending_test.clear()

            # 3. Transmission
            infected_all = set(time_I) | set(time_T)
            new_infected = {}
            for u, v, w in edges_by_snapshot[snap]:
                if u in infected_all and v not in infected_all and v not in new_infected:
                    if np.random.rand() < min(1.0, w * beta_current.get(u, 0.001)):
                        new_infected[v] = t
                if v in infected_all and u not in infected_all and u not in new_infected:
                    if np.random.rand() < min(1.0, w * beta_current.get(v, 0.001)):
                        new_infected[u] = t

            for node, t0 in new_infected.items():
                time_I[node]       = t0
                pending_test[node] = t0     # testing resolved at next snapshot
                ever_infected.add(node)

            # 4. Record counts
            counts_I[sim, t]    = len(time_I)
            counts_T[sim, t]    = len(time_T)
            counts_S[sim, t]    = len(all_nodes) - counts_I[sim, t] - counts_T[sim, t]
            counts_ever[sim, t] = len(ever_infected)

    return {'S': counts_S, 'I': counts_I, 'T': counts_T, 'ever': counts_ever}, snapshots


def run_SITS_simulations(df_contacts, beta_map,
                          n_simulations=100, n_seeds=10,
                          p_test=0.5,
                          recovery_I_years=1.0,
                          recovery_T_months=5,
                          beta_tested=0.001):
    """
    SITS simulation for the real-data network (Female_id / Male_id / Month_count format).

    Parameters
    ----------
    df_contacts       : DataFrame with Female_id, Male_id, Month_count columns
    beta_map          : dict  node_id -> original transmission rate
    p_test            : probability an infected node gets tested the next snapshot
    recovery_I_years  : I->S recovery time in years  (1 snapshot = 1 month)
    recovery_T_months : T->S recovery time in months (snapshots)
    beta_tested       : transmission rate for T nodes

    Returns
    -------
    counts    : dict {'S', 'I', 'T'}, each ndarray (n_simulations, n_snapshots)
    snapshots : list
    """
    all_nodes      = list(set(df_contacts['Female_id']) | set(df_contacts['Male_id']))
    edges_by_month = _build_weighted_edges(df_contacts)
    return _run_SITS_core(
        edges_by_month, all_nodes, beta_map,
        n_simulations, n_seeds, p_test,
        round(recovery_I_years * 12), recovery_T_months, beta_tested
    )


def run_SITS_simulations_generic(edges_by_snapshot, all_nodes, beta_map,
                                   n_simulations=100, n_seeds=10,
                                   p_test=0.5,
                                   recovery_I_years=1.0,
                                   recovery_T_months=5,
                                   beta_tested=0.001):
    """
    SITS simulation for a generic weighted temporal network.

    Parameters
    ----------
    edges_by_snapshot : dict  snapshot_key -> list of (u, v, weight)
    all_nodes         : list  all node ids
    beta_map          : dict  node_id -> original transmission rate
    p_test            : probability an infected node gets tested the next snapshot
    recovery_I_years  : I->S recovery time in years  (1 snapshot = 1 month)
    recovery_T_months : T->S recovery time in months (snapshots)
    beta_tested       : transmission rate for T nodes

    Returns
    -------
    counts    : dict {'S', 'I', 'T'}, each ndarray (n_simulations, n_snapshots)
    snapshots : list
    """
    return _run_SITS_core(
        edges_by_snapshot, all_nodes, beta_map,
        n_simulations, n_seeds, p_test,
        round(recovery_I_years * 12), recovery_T_months, beta_tested
    )


# ---------------------------------------------------------------------------
# Periodic-testing model (SIT_periodic)
# ---------------------------------------------------------------------------

def _run_SITp_core(edges_by_snapshot, all_nodes, beta_map,
                   n_simulations, n_seeds, T_t, recovery_T_snaps, beta_tested):
    """
    Periodic-testing epidemic model.

    Compartments
    ------------
    S  susceptible
    I  infected (original beta)
    T  infected, tested (beta_tested); recovers to S after recovery_T_snaps

    Testing schedule
    ----------------
    Each node's first test is drawn uniformly at random from [0, T_t) snapshots,
    then every T_t snapshots thereafter.
    At a scheduled test:
      - S nodes: unaffected
      - I nodes: move to T (beta drops to beta_tested)
      - T nodes: recovery clock resets (re-tested → recovery_T_snaps more snapshots to recovery)

    Recovery
    --------
    T → S after recovery_T_snaps snapshots; original beta restored.
    I nodes never recover on their own (absorbing unless tested).

    Returns
    -------
    counts    : dict {'S', 'I', 'T', 'ever'}, each ndarray (n_simulations, n_snapshots)
                'ever' counts cumulative unique nodes that have entered I or T at any point
    snapshots : list
    """
    snapshots = sorted(edges_by_snapshot.keys())
    n_snaps   = len(snapshots)
    all_nodes = list(all_nodes)

    counts_S    = np.zeros((n_simulations, n_snaps), dtype=int)
    counts_I    = np.zeros((n_simulations, n_snaps), dtype=int)
    counts_T    = np.zeros((n_simulations, n_snaps), dtype=int)
    counts_ever = np.zeros((n_simulations, n_snaps), dtype=int)

    for sim in range(n_simulations):
        # Staggered initial test offsets: uniform in [0, T_t)
        test_offset = {node: np.random.randint(0, T_t) for node in all_nodes}

        state_I       = {}   # node -> t when entered I
        state_T       = {}   # node -> t when moved to T (for recovery timing)
        beta_current  = dict(beta_map)
        ever_infected = set()

        # Seeds start as I from snapshot 0
        for node in np.random.choice(all_nodes, size=n_seeds, replace=False):
            state_I[node] = 0
            ever_infected.add(node)

        for t, snap in enumerate(snapshots):

            # 1. Recovery: T -> S after recovery_T_snaps
            for node in [n for n, t0 in list(state_T.items())
                         if (t - t0) >= recovery_T_snaps]:
                del state_T[node]
                beta_current[node] = beta_map[node]

            # 2. Periodic testing: check every node whose scheduled test falls on t
            offset = test_offset  # alias for clarity
            for node in list(state_I):   # only I nodes are affected by testing
                t0 = offset[node]
                if t >= t0 and (t - t0) % T_t == 0:
                    state_T[node] = t
                    del state_I[node]
                    beta_current[node] = beta_tested

            # Also re-test T nodes (resets recovery clock)
            for node in list(state_T):
                t0 = offset[node]
                if t >= t0 and (t - t0) % T_t == 0:
                    state_T[node] = t   # reset recovery clock

            # 3. Transmission
            infected_all = set(state_I) | set(state_T)
            new_infected = {}
            for u, v, w in edges_by_snapshot[snap]:
                if u in infected_all and v not in infected_all and v not in new_infected:
                    if np.random.rand() < min(1.0, w * beta_current.get(u, 0.001)):
                        new_infected[v] = t
                if v in infected_all and u not in infected_all and u not in new_infected:
                    if np.random.rand() < min(1.0, w * beta_current.get(v, 0.001)):
                        new_infected[u] = t

            for node, t0 in new_infected.items():
                state_I[node] = t0
                ever_infected.add(node)

            # 4. Record counts
            counts_I[sim, t]    = len(state_I)
            counts_T[sim, t]    = len(state_T)
            counts_S[sim, t]    = len(all_nodes) - counts_I[sim, t] - counts_T[sim, t]
            counts_ever[sim, t] = len(ever_infected)

    return {'S': counts_S, 'I': counts_I, 'T': counts_T, 'ever': counts_ever}, snapshots


def run_SITp_simulations(df_contacts, beta_map,
                          n_simulations=1000, n_seeds=10,
                          T_t=12, recovery_T_months=5, beta_tested=0.001):
    """
    Periodic-testing model on the real-data network (Female_id / Male_id format).

    Parameters
    ----------
    df_contacts       : DataFrame with Female_id, Male_id, Month_count columns
    beta_map          : dict  node_id -> original transmission rate
    T_t               : int   testing interval in months (snapshots)
    recovery_T_months : int   T->S recovery time in months after testing
    beta_tested       : float transmission rate for tested (T) nodes

    Returns
    -------
    counts    : dict {'S', 'I', 'T'}, each ndarray (n_simulations, n_snapshots)
    snapshots : list
    """
    all_nodes      = list(set(df_contacts['Female_id']) | set(df_contacts['Male_id']))
    edges_by_month = _build_weighted_edges(df_contacts)
    return _run_SITp_core(
        edges_by_month, all_nodes, beta_map,
        n_simulations, n_seeds, T_t, recovery_T_months, beta_tested
    )


def run_SITp_simulations_generic(edges_by_snapshot, all_nodes, beta_map,
                                   n_simulations=1000, n_seeds=10,
                                   T_t=12, recovery_T_months=5, beta_tested=0.001):
    """
    Periodic-testing model on a generic weighted temporal network.

    Parameters
    ----------
    edges_by_snapshot : dict  snapshot_key -> list of (u, v, weight)
    all_nodes         : list  all node ids
    beta_map          : dict  node_id -> original transmission rate
    T_t               : int   testing interval in months (snapshots)
    recovery_T_months : int   T->S recovery time in months after testing
    beta_tested       : float transmission rate for tested (T) nodes

    Returns
    -------
    counts    : dict {'S', 'I', 'T'}, each ndarray (n_simulations, n_snapshots)
    snapshots : list
    """
    return _run_SITp_core(
        edges_by_snapshot, all_nodes, beta_map,
        n_simulations, n_seeds, T_t, recovery_T_months, beta_tested
    )


def summarize_epidemic(counts, outbreak_threshold=50, last_n=12):
    """
    Outcome measures for an ensemble of runs returned by the SITS / SITp models.

    Parameters
    ----------
    counts             : dict with keys 'I', 'T', 'ever', each ndarray (n_simulations, n_snapshots)
    outbreak_threshold : int  a run counts as an outbreak if more than this many nodes
                              (seeds included) were ever infected
    last_n             : int  number of final snapshots over which prevalence is averaged

    Returns
    -------
    dict with
        'ever'       : ndarray (n_simulations,)  cumulative incidence (nodes ever infected)
        'prevalence' : ndarray (n_simulations,)  mean I+T over the last last_n snapshots
        'outbreak'   : bool ndarray (n_simulations,)  ever > outbreak_threshold
        'p_outbreak' : float  fraction of runs that are outbreaks
    """
    ever       = counts['ever'][:, -1]
    prevalence = (counts['I'] + counts['T'])[:, -last_n:].mean(axis=1)
    outbreak   = ever > outbreak_threshold
    return {
        'ever':       ever,
        'prevalence': prevalence,
        'outbreak':   outbreak,
        'p_outbreak': outbreak.mean(),
    }
