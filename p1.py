
import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from scipy.integrate import simpson
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

w_b = 1.0           # Qubit transition frequency
g = 0.1             # Coupling strength (Increased to enter Strong Coupling / Non-Markovian regime)
gamma0_base = 0.05  # Base cavity decay rate (Decreased to allow information backflow)
lam_decay = 0.05    # Non-Markovian memory scale (Env A)
n0 = 0.1             # Thermal photon number (Env B)
gamma_phi = 0.02    # Pure dephasing rate (Env B)
N_cavity = 6        # Truncated Fock space size
tlist = np.linspace(0, 20, 150) # Simulation time
Omega= np.pi / 5.0
def simulate_battery(N_qubits, env_type, delta, current_gamma=gamma0_base):
    Jz = tensor(jmat(N_qubits/2, 'z'), qeye(N_cavity))
    Jp = tensor(jmat(N_qubits/2, '+'), qeye(N_cavity))
    Jm = tensor(jmat(N_qubits/2, '-'), qeye(N_cavity))
    a  = tensor(qeye(N_qubits + 1), destroy(N_cavity))
    H_B_total = w_b * (Jz + N_qubits/2)
    
    psi_charged = tensor(basis(N_qubits+1, 0), basis(N_cavity, 0)) 
    psi_empty = tensor(basis(N_qubits+1, N_qubits), basis(N_cavity, 0)) 
    rho1_init, rho2_init = ket2dm(psi_charged), ket2dm(psi_empty)
    
    H = H_B_total + (w_b + delta) * a.dag() * a + g * (Jp * a + Jm * a.dag())
    H_evals = np.sort(jmat(N_qubits/2, 'z').eigenenergies() + N_qubits/2) * w_b

    def calc_ergo(state):
        rho_b = state.ptrace(0)
        evals = np.sort(rho_b.eigenenergies())[::-1]
        E_pass = np.sum(evals * H_evals)
        return max(0.0, expect(H_B_total.ptrace(0), rho_b) - E_pass)

    c_ops = []
    args_dict = {}
    
    if env_type == 'A':
        def decay(t, args): 
            return np.sqrt(args['gamma'] * np.exp(-lam_decay * t))
        c_ops.append([a, decay])
        args_dict = {'gamma': current_gamma}
        
    elif env_type == 'B':
        def gamma_emission(t, args):
            n_th_t = args['n0'] * (1 + np.sin(args['Omega'] * t)**2)
            return np.sqrt(args['gamma'] * (1 + n_th_t))
        
        def gamma_absorption(t, args):
            n_th_t = args['n0'] * (1 + np.sin(args['Omega'] * t)**2)
            return np.sqrt(args['gamma'] * n_th_t)

        c_ops.append([a, gamma_emission])
        c_ops.append([a.dag(), gamma_absorption])
        c_ops.append(np.sqrt(gamma_phi) * Jz)
        args_dict = {'gamma': current_gamma, 'n0': n0, 'Omega': Omega}

    res1 = mesolve(H, rho1_init, tlist, c_ops, [], args=args_dict)
    res2 = mesolve(H, rho2_init, tlist, c_ops, [], args=args_dict)
    
    ergo_dynamics = [calc_ergo(s) for s in res1.states]
    dist = [tracedist(res1.states[i].ptrace(0), res2.states[i].ptrace(0)) for i in range(len(tlist))]
    derivs = np.gradient(dist, tlist[1]-tlist[0])
    blp = simpson(np.maximum(derivs, 0), x=tlist)
    
    return np.array(ergo_dynamics), np.array(dist), blp

print("start")
start_time = time.time()

results = []
plot_data = {}

for N in [1, 2, 3, 4]:
    print(f".")
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
        gain_ratio = eopt_val / e0_val if e0_val > 0 else 0
        gain_pct = ((eopt_val - e0_val) / e0_val) * 100 if e0_val > 0 else 0
        
        results.append({
            'N': N, 'Env': env, 'Delta_opt': round(delta_opt, 4),
            'E_res_0': round(e0_val, 4), 'E_res_opt': round(eopt_val, 4),
            'Gain_Ratio (x)': round(gain_ratio, 2),
            'Gain_Percent (%)': round(gain_pct, 2),
            'BLP_0': round(blp_0, 4), 'BLP_opt': round(blp_opt, 4)
        })

print("\n" + "="*110)
print(f"| {'N':^2} | {'Env':^3} | {'Delta*':^8} | {'E_0 (Unfiltered)':^18} | {'E_opt (Filtered)':^18} | {'Gain (%)':^10} | {'BLP_0':^8} | {'BLP_opt':^8} |")
print("-" * 110)
for r in results:
    print(f"| {r['N']:^2} | {r['Env']:^3} | {r['Delta_opt']:^8.4f} | {r['E_res_0']:^18.4f} | {r['E_res_opt']:^18.4f} | {r['Gain_Percent (%)']:^10.2f} | {r['BLP_0']:^8.4f} | {r['BLP_opt']:^8.4f} |")
print("="*110 + "\n")

csv_file = "Comprehensive_Battery_Results_V2.csv"
with open(csv_file, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

env_names = {'A': 'Env A', 'B': 'Env B'}

fig1, axes = plt.subplots(4, 2, figsize=(12, 16), sharex=True)
for idx, N in enumerate([1, 2, 3, 4]):
    for j, env in enumerate(['A', 'B']):
        ax = axes[idx, j]
        d = plot_data[N][env]
        ax.plot(tlist, d['e0'], 'k--')
        ax.plot(tlist, d['e_opt'], 'r-')
        ax.set_title(f"({chr(97 + idx*2 + j)}) $N={N}$ | {env_names[env]}")
        ax.set_ylabel(r'$\mathcal{E}(t)$')
        ax.grid(alpha=0.3)
        if idx == 3: ax.set_xlabel(r'Time $t$')
        if idx == 0 and j == 0: ax.legend()
plt.tight_layout()
plt.savefig('Fig1_Master_Dynamics.png')
plt.close()

fig2, axes = plt.subplots(3, 2, figsize=(12, 14))
delta_scan = np.linspace(0, 0.5, 15)
gamma_scan = np.linspace(0.02, 0.15, 15)

for idx, N in enumerate([2, 3, 4]):
    for j, env in enumerate(['A', 'B']):
        ax = axes[idx, j]
        map_data = np.zeros((len(gamma_scan), len(delta_scan)))
        
        for iy, g0_val in enumerate(gamma_scan):
            for ix, dlt in enumerate(delta_scan):
                e, _, _ = simulate_battery(N, env, dlt, current_gamma=g0_val)
                map_data[iy, ix] = e[-1]
                
        X, Y = np.meshgrid(delta_scan, gamma_scan)
        cp = ax.contourf(X, Y, map_data, levels=30, cmap='magma')
        fig2.colorbar(cp, ax=ax, label=r'$\mathcal{E}_{res}$')
        ax.set_title(f"({chr(97 + idx*2 + j)}) $N={N}$ | {env_names[env]}")
        ax.set_xlabel(r'Detuning $\Delta$')
        ax.set_ylabel(r'Base Decay $\gamma_0$')
        
        theo_opt = g * np.sqrt(N / (2 * gamma_scan))
        ax.plot(theo_opt, gamma_scan, 'w--', lw=2, label='Analytical $\Delta^*$')
        ax.set_xlim(0, 0.5)
        if idx==0 and j==0: ax.legend()

plt.tight_layout()
plt.savefig('Fig2_Survival_Heatmaps.png')
plt.close()

fig3 = plt.figure(figsize=(15, 10))
gs = fig3.add_gridspec(2, 2)

ax3a = fig3.add_subplot(gs[0, 0])
N_arr = np.array([1, 2, 3, 4])
d_opts_A = [plot_data[n]['A']['delta'] for n in N_arr]
ax3a.plot(N_arr, g*np.sqrt(N_arr/(2*gamma0_base)), 'k--', label=r'$\propto \sqrt{N}$')
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
    e, _, b = simulate_battery(2, 'A', d)
    ergo_p.append(e[-1]); blp_p.append(b)

color = 'tab:red'
ax3b.plot(d_scan, ergo_p, 'ro-', label=r'$\mathcal{E}_{res}$')
ax3b.set_ylabel(r'$\mathcal{E}_{res}$', color=color)
ax3b.tick_params(axis='y', labelcolor=color)

ax3b_twin = ax3b.twinx()
color = 'tab:blue'
ax3b_twin.plot(d_scan, blp_p, 'bs--', label=r'BLP $\mathcal{N}$')
ax3b_twin.set_ylabel(r'BLP $\mathcal{N}$', color=color)
ax3b_twin.tick_params(axis='y', labelcolor=color)
ax3b.set_title('(b) The Non-Markovian Paradox ($N=2$)')
ax3b.set_xlabel(r'$\Delta$')
ax3b.axvline(plot_data[2]['A']['delta'], color='k', ls=':', label=r'$\Delta^*$')
ax3b.grid(alpha=0.3)

ax3c = fig3.add_subplot(gs[1, :])
adv_A = [plot_data[n]['A']['e_opt'][-1] / (n * plot_data[1]['A']['e_opt'][-1]) for n in N_arr]
adv_B = [plot_data[n]['B']['e_opt'][-1] / (n * plot_data[1]['B']['e_opt'][-1]) for n in N_arr]

ax3c.plot(N_arr, adv_A, 'bo-', label='Env A (Nuclear Decay)')
ax3c.plot(N_arr, adv_B, 'rs--', label='Env B (Thermal)')
ax3c.axhline(1.0, color='k', ls='--')
ax3c.fill_between(N_arr, 1.0, adv_A, where=(np.array(adv_A)>=1.0), color='blue', alpha=0.1)
ax3c.set_title('(c) Quantum Advantage Metric')
ax3c.set_xlabel('$N$')
ax3c.set_ylabel(r'$\mathcal{E}_N / (N \times \mathcal{E}_1)$')
ax3c.set_xticks(N_arr)
ax3c.legend()
ax3c.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('Fig3_Scaling_and_Paradox.png')
plt.close()

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
plt.savefig('Fig4_RWA_Breakdown.png')
plt.close()
#beta
from scipy.optimize import curve_fit

print("\n" + "="*50)
print("="*50)

def power_law(N, A, beta):
    return A * np.power(N, beta)

N_array = np.array([1, 2, 3, 4])

delta_fine_scan = np.linspace(0.05, 0.4, 50)
numerical_delta_opt_A = []
numerical_delta_opt_B = []

for N in N_array:
    ergo_peaks_A = []
    ergo_peaks_B = []
    for dlt in delta_fine_scan:
        e_A, _, _ = simulate_battery(N, 'A', dlt)
        e_B, _, _ = simulate_battery(N, 'B', dlt)
        ergo_peaks_A.append(e_A[-1])
        ergo_peaks_B.append(e_B[-1])
    
    numerical_delta_opt_A.append(delta_fine_scan[np.argmax(ergo_peaks_A)])
    numerical_delta_opt_B.append(delta_fine_scan[np.argmax(ergo_peaks_B)])

popt_A, _ = curve_fit(power_law, N_array, numerical_delta_opt_A)
beta_A = popt_A[1]

popt_B, _ = curve_fit(power_law, N_array, numerical_delta_opt_B)
beta_B = popt_B[1]

print(f" Analytical Beta       = 0.5000")
print(f" Empirical Beta (Env A) = {beta_A:.4f}")
print(f" Empirical Beta (Env B) = {beta_B:.4f}")
print("="*50)

print(f" Done! Execution time: {(time.time() - start_time):.2f} seconds.")
