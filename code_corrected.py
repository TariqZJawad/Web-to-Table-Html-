"""
Corrected simulation code for "Ergotropy Protection via Cavity Detuning in
Collective Open Quantum Batteries".

Fixes applied relative to the original code.py (see accompanying audit notes):

  FIX 1 (critical): calc_ergo previously computed Tr(rho_B H_B) via
  `expect(H_B_total.ptrace(0), rho_b)`, where H_B_total = w_b*(Jz(x)I_cavity + N/2).
  Partial-tracing an operator of the form A(x)I over the traced-out subsystem
  returns A*dim(traced subsystem), NOT A. This silently inflated the first term
  of the ergotropy formula by a factor of N_cavity=6 while leaving the second
  (passive-state) term unscaled -- confirmed empirically: the original Fig. 1
  curves start at exactly 6*N (6, 12, 18, 24 for N=1,2,3,4) instead of the
  physically correct N. Fixed by defining a battery-only Hamiltonian operator
  independent of ptrace.

  FIX 2: Environment A is now implemented as genuine random telegraph noise
  (RTN) on the battery transition frequency, omega_b(t) = w_b + delta_fluct*chi(t),
  chi(t) in {+1,-1} switching at Poisson rate lam_decay, ensemble-averaged over
  N_TRAJ realizations -- matching the RTN description in the manuscript text
  (the original code instead used an ad hoc deterministic exponentially-decaying
  cavity-loss rate, which is neither RTN nor consistent with any of the three
  conflicting descriptions found across the manuscript/code/figure legend).
  A constant-rate cavity-loss channel (rate gamma_0) is kept in parallel, since
  this is the channel that detuning Delta is protecting against (the superradiant
  channel); without it, Delta would have nothing to protect in Environment A.
  NOTE: the fluctuation amplitude delta_fluct is not numerically specified
  anywhere in the manuscript. We set delta_fluct = g (the only other available
  energy scale). Confirm/tune this before treating results as final.

  FIX 3: gamma_0 is now used as the single, consistent cavity-linewidth
  parameter (previously conflated in the manuscript text with a separately
  named kappa and with an unrelated Ohmic-bath cutoff also confusingly called
  omega_c). No change to Environment B's structure was needed (it already
  matched the text and used gamma_0 consistently); only the ergotropy fix
  (FIX 1) applies to it.

Run this end-to-end to regenerate Table 1, Fig. 1-4, and the fitted beta
exponents with the corrected physics.
"""

import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from scipy.integrate import simpson
from scipy.optimize import curve_fit
import csv
import time

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'lines.linewidth': 2.0,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
w_b = 1.0            # Qubit transition frequency
g = 0.1              # Battery-cavity coupling strength
gamma0_base = 0.05   # Cavity linewidth / "standard decay rate" (== kappa)
lam_decay = 0.05     # RTN switching (Poisson) rate for Environment A
delta_fluct = g      # RTN frequency-fluctuation amplitude (ASSUMPTION -- see header)
n0 = 0.1             # Thermal photon number (Env B)
gamma_phi = 0.02     # Pure dephasing rate (Env B)
N_cavity = 6         # Truncated Fock space size
tlist = np.linspace(0, 20, 150)
Omega = np.pi / 5.0
N_TRAJ = 60          # RTN trajectory-averaging count, used UNIFORMLY everywhere in this
                     # script (no reduced counts anywhere). Correctness over runtime, per
                     # explicit instruction -- raise further if you want tighter statistics
                     # and can afford the runtime; there is no longer any code path that
                     # silently uses fewer trajectories.

RNG = np.random.default_rng(12345)  # fixed seed for reproducibility


def generate_rtn_trajectory(t_grid, lam, rng):
    """Sample one realization of chi(t) in {+1,-1}, Poisson switching at rate lam."""
    chi = np.empty(len(t_grid))
    state = 1.0 if rng.random() < 0.5 else -1.0
    next_switch = t_grid[0] + rng.exponential(1.0 / lam)
    for i, t in enumerate(t_grid):
        while next_switch <= t:
            state *= -1.0
            next_switch += rng.exponential(1.0 / lam)
        chi[i] = state
    return chi


def simulate_battery(N_qubits, env_type, delta, current_gamma=gamma0_base, n_traj=N_TRAJ):
    Jz = tensor(jmat(N_qubits / 2, 'z'), qeye(N_cavity))
    Jp = tensor(jmat(N_qubits / 2, '+'), qeye(N_cavity))
    Jm = tensor(jmat(N_qubits / 2, '-'), qeye(N_cavity))
    a = tensor(qeye(N_qubits + 1), destroy(N_cavity))
    Jz_battery_shift = tensor(jmat(N_qubits / 2, 'z') + (N_qubits / 2) * qeye(N_qubits + 1),
                               qeye(N_cavity))  # operator multiplying the RTN fluctuation

    # FIX 1: battery-only Hamiltonian, defined independently of any partial trace.
    H_B_battery = w_b * (jmat(N_qubits / 2, 'z') + N_qubits / 2)
    H_evals = np.sort(H_B_battery.eigenenergies())

    H_B_total = w_b * (Jz + N_qubits / 2)
    H_static = H_B_total + (w_b + delta) * a.dag() * a + g * (Jp * a + Jm * a.dag())

    psi_charged = tensor(basis(N_qubits + 1, 0), basis(N_cavity, 0))
    psi_empty = tensor(basis(N_qubits + 1, N_qubits), basis(N_cavity, 0))
    rho1_init, rho2_init = ket2dm(psi_charged), ket2dm(psi_empty)

    def calc_ergo_from_reduced(rho_b):
        """rho_b is already the battery-only (ptrace'd) reduced density matrix."""
        evals = np.sort(rho_b.eigenenergies())[::-1]
        E_pass = np.sum(evals * H_evals)
        return max(0.0, expect(H_B_battery, rho_b) - E_pass)

    def calc_ergo(state):
        """state is the full joint (battery+cavity) density matrix."""
        return calc_ergo_from_reduced(state.ptrace(0))

    if env_type == 'A':
        # FIX 2: genuine RTN on the battery frequency + constant-rate cavity loss.
        # Physically, the unconditional (unmeasured-noise) state at time t is the
        # ENSEMBLE AVERAGE of the reduced density matrices over chi(t) realizations,
        # rho_bar(t) = <rho_chi(t)>. Ergotropy and trace distance are both nonlinear
        # functionals of rho, so we must average the density matrices themselves
        # first and evaluate calc_ergo / tracedist once on the averaged states --
        # NOT average the per-trajectory scalar ergotropy/trace-distance values.
        c_ops = [np.sqrt(current_gamma) * a]
        rho1_accum = [None for _ in tlist]
        rho2_accum = [None for _ in tlist]
        for _ in range(n_traj):
            chi_traj = generate_rtn_trajectory(tlist, lam_decay, RNG)

            def chi_coeff(t, _traj=chi_traj):
                return delta_fluct * np.interp(t, tlist, _traj)

            H = [H_static, [Jz_battery_shift, chi_coeff]]
            res1 = mesolve(H, rho1_init, tlist, c_ops=c_ops, e_ops=[])
            res2 = mesolve(H, rho2_init, tlist, c_ops=c_ops, e_ops=[])
            for i in range(len(tlist)):
                rb1 = res1.states[i].ptrace(0)
                rb2 = res2.states[i].ptrace(0)
                rho1_accum[i] = rb1 if rho1_accum[i] is None else rho1_accum[i] + rb1
                rho2_accum[i] = rb2 if rho2_accum[i] is None else rho2_accum[i] + rb2

        rho1_avg = [r / n_traj for r in rho1_accum]
        rho2_avg = [r / n_traj for r in rho2_accum]
        ergo_dynamics = np.array([calc_ergo_from_reduced(r) for r in rho1_avg])
        dist = np.array([tracedist(rho1_avg[i], rho2_avg[i]) for i in range(len(tlist))])

    elif env_type == 'B':
        def gamma_emission(t):
            n_th_t = n0 * (1 + np.sin(Omega * t) ** 2)
            return np.sqrt(current_gamma * (1 + n_th_t))

        def gamma_absorption(t):
            n_th_t = n0 * (1 + np.sin(Omega * t) ** 2)
            return np.sqrt(current_gamma * n_th_t)

        c_ops = [[a, gamma_emission], [a.dag(), gamma_absorption], np.sqrt(gamma_phi) * Jz]
        res1 = mesolve(H_static, rho1_init, tlist, c_ops=c_ops, e_ops=[])
        res2 = mesolve(H_static, rho2_init, tlist, c_ops=c_ops, e_ops=[])
        ergo_dynamics = np.array([calc_ergo(s) for s in res1.states])
        dist = np.array([tracedist(res1.states[i].ptrace(0), res2.states[i].ptrace(0))
                          for i in range(len(tlist))])

    else:
        raise ValueError("env_type must be 'A' or 'B'")

    derivs = np.gradient(dist, tlist[1] - tlist[0])
    blp = simpson(np.maximum(derivs, 0), x=tlist)

    return np.array(ergo_dynamics), np.array(dist), blp


# ---------------------------------------------------------------------------
# Main sweep: Table 1 + Fig. 1 data
# ---------------------------------------------------------------------------
print("start")
start_time = time.time()

results = []
plot_data = {}

for N in [1, 2, 3, 4]:
    print(f"N={N}")
    plot_data[N] = {}
    delta_opt = g * np.sqrt(N / (2 * gamma0_base))

    for env in ['A', 'B']:
        ergo_0, dist_0, blp_0 = simulate_battery(N, env, 0.0)
        ergo_opt, dist_opt, blp_opt = simulate_battery(N, env, delta_opt)

        plot_data[N][env] = {
            'e0': ergo_0, 'e_opt': ergo_opt,
            'd0': dist_0, 'd_opt': dist_opt,
            'b0': blp_0, 'b_opt': blp_opt,
            'delta': delta_opt
        }

        e0_val = ergo_0[-1]
        eopt_val = ergo_opt[-1]
        abs_gain = eopt_val - e0_val
        BASELINE_FLOOR = 1e-3  # below this, a percentage gain is not a meaningful quantity
        if e0_val > BASELINE_FLOOR:
            gain_ratio = eopt_val / e0_val
            gain_pct = (abs_gain / e0_val) * 100
        else:
            gain_ratio = np.nan
            gain_pct = np.nan

        results.append({
            'N': N, 'Env': env, 'Delta_opt': round(delta_opt, 4),
            'E_res_0': round(e0_val, 4), 'E_res_opt': round(eopt_val, 4),
            'Abs_Gain': round(abs_gain, 4),
            'Gain_Ratio (x)': round(gain_ratio, 2) if not np.isnan(gain_ratio) else 'N/A (baseline~0)',
            'Gain_Percent (%)': round(gain_pct, 2) if not np.isnan(gain_pct) else 'N/A (baseline~0)',
            'BLP_0': round(blp_0, 4), 'BLP_opt': round(blp_opt, 4)
        })

print("\n" + "=" * 120)
print(f"| {'N':^2} | {'Env':^3} | {'Delta*':^8} | {'E_0 (Unfilt)':^13} | {'E_opt (Filt)':^13} | {'Abs Gain':^9} | {'Gain (%)':^18} | {'BLP_0':^8} | {'BLP_opt':^8} |")
print("-" * 120)
for r in results:
    gp = r['Gain_Percent (%)']
    gp_str = f"{gp:>18.2f}" if isinstance(gp, (int, float)) else f"{gp:^18}"
    print(f"| {r['N']:^2} | {r['Env']:^3} | {r['Delta_opt']:^8.4f} | {r['E_res_0']:^13.4f} | {r['E_res_opt']:^13.4f} | {r['Abs_Gain']:^9.4f} | {gp_str} | {r['BLP_0']:^8.4f} | {r['BLP_opt']:^8.4f} |")
print("=" * 110 + "\n")

csv_file = "Comprehensive_Battery_Results_Corrected.csv"
with open(csv_file, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

env_names = {'A': 'Env A', 'B': 'Env B'}

# ---------------------------------------------------------------------------
# Fig 1: joint ergotropy / BLP dynamics
# ---------------------------------------------------------------------------
fig1, axes = plt.subplots(4, 2, figsize=(12, 16), sharex=True)
for idx, N in enumerate([1, 2, 3, 4]):
    for j, env in enumerate(['A', 'B']):
        ax = axes[idx, j]
        d = plot_data[N][env]
        ax.plot(tlist, d['e0'], 'k--', label=r'$\Delta=0$')
        ax.plot(tlist, d['e_opt'], 'r-', label=r'$\Delta=\Delta^*$')
        ax.set_title(f"({chr(97 + idx * 2 + j)}) $N={N}$ | {env_names[env]}")
        ax.set_ylabel(r'$\mathcal{E}(t)$')
        ax.grid(alpha=0.3)
        if idx == 3:
            ax.set_xlabel(r'Time $t$')
        if idx == 0 and j == 0:
            ax.legend()
plt.tight_layout()
plt.savefig('Fig1_Master_Dynamics_Corrected.png')
plt.close()

# ---------------------------------------------------------------------------
# Fig 2: survival maps (uses N_TRAJ uniformly, same as everywhere else --
# this grid is 15x15x3xN_TRAJ mesolve pairs for Env A, expect a long runtime)
# ---------------------------------------------------------------------------
fig2, axes = plt.subplots(3, 2, figsize=(12, 14))
delta_scan = np.linspace(0, 0.5, 15)
gamma_scan = np.linspace(0.02, 0.15, 15)

for idx, N in enumerate([2, 3, 4]):
    for j, env in enumerate(['A', 'B']):
        ax = axes[idx, j]
        map_data = np.zeros((len(gamma_scan), len(delta_scan)))

        for iy, g0_val in enumerate(gamma_scan):
            for ix, dlt in enumerate(delta_scan):
                e, _, _ = simulate_battery(N, env, dlt, current_gamma=g0_val, n_traj=N_TRAJ)
                map_data[iy, ix] = e[-1]

        X, Y = np.meshgrid(delta_scan, gamma_scan)
        cp = ax.contourf(X, Y, map_data, levels=30, cmap='magma')
        fig2.colorbar(cp, ax=ax, label=r'$\mathcal{E}_{res}$')
        ax.set_title(f"({chr(97 + idx * 2 + j)}) $N={N}$ | {env_names[env]}")
        ax.set_xlabel(r'Detuning $\Delta$')
        ax.set_ylabel(r'Base Decay $\gamma_0$')

        theo_opt = g * np.sqrt(N / (2 * gamma_scan))
        ax.plot(theo_opt, gamma_scan, 'w--', lw=2, label='Analytical $\\Delta^*$')
        ax.set_xlim(0, 0.5)
        if idx == 0 and j == 0:
            ax.legend()

plt.tight_layout()
plt.savefig('Fig2_Survival_Heatmaps_Corrected.png')
plt.close()

# ---------------------------------------------------------------------------
# Fig 3: scaling law, non-Markovian trade-off, quantum advantage
# ---------------------------------------------------------------------------
fig3 = plt.figure(figsize=(15, 10))
gs = fig3.add_gridspec(2, 2)

ax3a = fig3.add_subplot(gs[0, 0])
N_arr = np.array([1, 2, 3, 4])
ax3a.plot(N_arr, g * np.sqrt(N_arr / (2 * gamma0_base)), 'k--', label=r'$\propto \sqrt{N}$')
ax3a.set_title('(a) Scaling of Optimal Detuning')
ax3a.set_xlabel('$N$')
ax3a.set_ylabel(r'$\Delta^*$')
ax3a.set_xticks(N_arr)
ax3a.legend()
ax3a.grid(alpha=0.3)

ax3b = fig3.add_subplot(gs[0, 1])
d_scan = np.linspace(0, 0.6, 15)
ergo_p, blp_p = [], []
for d in d_scan:
    e, _, b = simulate_battery(2, 'A', d, n_traj=N_TRAJ)
    ergo_p.append(e[-1])
    blp_p.append(b)

color = 'tab:red'
ax3b.plot(d_scan, ergo_p, 'ro-', label=r'$\mathcal{E}_{res}$')
ax3b.set_ylabel(r'$\mathcal{E}_{res}$', color=color)
ax3b.tick_params(axis='y', labelcolor=color)

ax3b_twin = ax3b.twinx()
color = 'tab:blue'
ax3b_twin.plot(d_scan, blp_p, 'bs--', label=r'BLP $\mathcal{N}$')
ax3b_twin.set_ylabel(r'BLP $\mathcal{N}$', color=color)
ax3b_twin.tick_params(axis='y', labelcolor=color)
ax3b.set_title('(b) Non-Markovian Trade-off ($N=2$)')
ax3b.set_xlabel(r'$\Delta$')
ax3b.axvline(plot_data[2]['A']['delta'], color='k', ls=':', label=r'$\Delta^*$')
ax3b.grid(alpha=0.3)

ax3c = fig3.add_subplot(gs[1, :])
adv_A = [plot_data[n]['A']['e_opt'][-1] / (n * plot_data[1]['A']['e_opt'][-1]) for n in N_arr]
adv_B = [plot_data[n]['B']['e_opt'][-1] / (n * plot_data[1]['B']['e_opt'][-1]) for n in N_arr]

ax3c.plot(N_arr, adv_A, 'bo-', label='Env A (RTN + cavity loss)')
ax3c.plot(N_arr, adv_B, 'rs--', label='Env B (Thermal)')
ax3c.axhline(1.0, color='k', ls='--')
ax3c.fill_between(N_arr, 1.0, adv_A, where=(np.array(adv_A) >= 1.0), color='blue', alpha=0.1)
ax3c.set_title('(c) Quantum Advantage Metric')
ax3c.set_xlabel('$N$')
ax3c.set_ylabel(r'$\mathcal{E}_N / (N \times \mathcal{E}_1)$')
ax3c.set_xticks(N_arr)
ax3c.legend()
ax3c.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('Fig3_Scaling_and_Tradeoff_Corrected.png')
plt.close()

# ---------------------------------------------------------------------------
# Fig 4: RWA breakdown (unaffected by the ergotropy fix -- unchanged, regenerated for completeness)
# ---------------------------------------------------------------------------
fig4, ax4 = plt.subplots(figsize=(8, 5))
N_ext = np.arange(1, 51)
g_eff = g * np.sqrt(N_ext)
ratio = g_eff / w_b

ax4.plot(N_ext, ratio, 'k-', lw=2.5)
ax4.axhline(0.1, color='r', linestyle='--', label='RWA Breakdown Threshold (~10%)')
ax4.fill_between(N_ext, 0.1, ratio, where=(ratio >= 0.1), color='red', alpha=0.2, label='USC Regime (Model Fails)')
ax4.set_title('Regime of Validity: Breakdown of Rotating Wave Approximation')
ax4.set_xlabel('Number of Qubits ($N$)')
ax4.set_ylabel(r'Coupling Ratio $g_{eff} / \omega_b$')
ax4.grid(alpha=0.3)
ax4.legend(loc='upper left')
plt.tight_layout()
plt.savefig('Fig4_RWA_Breakdown_Corrected.png')
plt.close()

# ---------------------------------------------------------------------------
# Beta exponent fitting
# ---------------------------------------------------------------------------
print("\n" + "=" * 50)


def power_law(N, A, beta):
    return A * np.power(N, beta)


def refine_peak_parabolic(x, y, i_max):
    """Sub-grid-resolution peak location via a local parabolic (3-point) fit
    around the discrete argmax, to reduce sensitivity of beta to residual
    RTN sampling noise in individual y[i] values. Falls back to the raw
    grid point at the scan boundary."""
    if i_max == 0 or i_max == len(x) - 1:
        return x[i_max]
    x0, x1, x2 = x[i_max - 1], x[i_max], x[i_max + 1]
    y0, y1, y2 = y[i_max - 1], y[i_max], y[i_max + 1]
    denom = (y0 - 2 * y1 + y2)
    if abs(denom) < 1e-12:
        return x1
    x_peak = x1 + 0.5 * (y0 - y2) / denom * (x1 - x0)
    # guard against the parabola extrapolating outside the local bracket
    return x_peak if x0 <= x_peak <= x2 else x1


N_array = np.array([1, 2, 3, 4])
numerical_delta_opt_A = []
numerical_delta_opt_B = []

for N in N_array:
    # ADAPTIVE SCAN RANGE (bug fix): the previous fixed range [0.05, 0.5] did not
    # even cover the analytical Delta*(N) for N=3 (0.5477) and N=4 (0.6325),
    # guaranteeing the numerical argmax was pinned at the scan boundary for those
    # N -- which mechanically forces beta -> 0 regardless of the true physics.
    # Each N now gets its own scan that brackets its analytical prediction with
    # comfortable margin (2x), so a genuine interior optimum, a monotonic curve,
    # or a boundary-pinned result can all be distinguished from a scan-range artifact.
    delta_analytical_N = g * np.sqrt(N / (2 * gamma0_base))
    delta_fine_scan = np.linspace(0.02, 2.0 * delta_analytical_N, 40)

    ergo_peaks_A = []
    ergo_peaks_B = []
    for dlt in delta_fine_scan:
        e_A, _, _ = simulate_battery(N, 'A', dlt, n_traj=N_TRAJ)
        e_B, _, _ = simulate_battery(N, 'B', dlt)
        ergo_peaks_A.append(e_A[-1])
        ergo_peaks_B.append(e_B[-1])

    i_max_A = int(np.argmax(ergo_peaks_A))
    i_max_B = int(np.argmax(ergo_peaks_B))

    # Flag (don't silently accept) if the discovered optimum is still pinned at
    # either boundary of its own adaptive scan -- this means even 2x the analytical
    # prediction wasn't enough range, and should be reported, not hidden.
    if i_max_A in (0, len(delta_fine_scan) - 1):
        print(f"  [WARNING] Env A, N={N}: numerical optimum pinned at scan boundary "
              f"(Delta={delta_fine_scan[i_max_A]:.4f}); range may still be insufficient.")
    if i_max_B in (0, len(delta_fine_scan) - 1):
        print(f"  [WARNING] Env B, N={N}: numerical optimum pinned at scan boundary "
              f"(Delta={delta_fine_scan[i_max_B]:.4f}); range may still be insufficient.")

    numerical_delta_opt_A.append(refine_peak_parabolic(delta_fine_scan, ergo_peaks_A, i_max_A))
    numerical_delta_opt_B.append(refine_peak_parabolic(delta_fine_scan, ergo_peaks_B, i_max_B))

popt_A, _ = curve_fit(power_law, N_array, numerical_delta_opt_A)
beta_A = popt_A[1]

popt_B, _ = curve_fit(power_law, N_array, numerical_delta_opt_B)
beta_B = popt_B[1]

print(f" Numerically found Delta_opt (Env A): {[round(x,4) for x in numerical_delta_opt_A]}")
print(f" Numerically found Delta_opt (Env B): {[round(x,4) for x in numerical_delta_opt_B]}")
print(f" Analytical Delta*(N) for comparison: {[round(g*np.sqrt(n/(2*gamma0_base)),4) for n in N_array]}")
print(f" Analytical Beta        = 0.5000")
print(f" Empirical Beta (Env A) = {beta_A:.4f}")
print(f" Empirical Beta (Env B) = {beta_B:.4f}")
print("=" * 50)

print(f" Done! Execution time: {(time.time() - start_time):.2f} seconds.")
