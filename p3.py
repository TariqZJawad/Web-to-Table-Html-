"""
==============================================================================
Paper I: Fast & Optimized Beyond-RWA CMM Quantum Battery Simulation
==============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from scipy.integrate import simpson
from scipy.sparse.linalg import eigs  
import csv
import time
import os

# ==========================================
# 1. Publication Quality Settings
# ==========================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'lines.linewidth': 2.0,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# ==========================================
# 2. System Parameters
# ==========================================
Nc = 4  
Nm = 3  
Np = 3  
Nq = 2  

w_c = 1.0        
w_m = 1.0        
w_p = 0.1        
w_q = 1.0        

lam = 0.2        
g_a = 0.15       

kappa_c = 0.05
kappa_m = 0.05
kappa_p = 0.01
gamma_q = 0.02

tlist = np.linspace(0, 40, 150) 
gb_scan = np.linspace(0.0, 0.2, 10)
delta_scan = np.linspace(-0.5, 0.5, 10)

# ==========================================
# 3. Operators Construction
# ==========================================
def build_operators():
    a  = tensor(destroy(Nc), qeye(Nm), qeye(Np), qeye(2), qeye(2))
    b  = tensor(qeye(Nc), destroy(Nm), qeye(Np), qeye(2), qeye(2))
    p  = tensor(qeye(Nc), qeye(Nm), destroy(Np), qeye(2), qeye(2))
    
    sm1 = tensor(qeye(Nc), qeye(Nm), qeye(Np), destroy(2), qeye(2))
    sm2 = tensor(qeye(Nc), qeye(Nm), qeye(Np), qeye(2), destroy(2))
    sz1 = tensor(qeye(Nc), qeye(Nm), qeye(Np), sigmaz(), qeye(2))
    sz2 = tensor(qeye(Nc), qeye(Nm), qeye(Np), qeye(2), sigmaz())
    sx1 = tensor(qeye(Nc), qeye(Nm), qeye(Np), sigmax(), qeye(2))
    sx2 = tensor(qeye(Nc), qeye(Nm), qeye(Np), qeye(2), sigmax())
    
    Jz = 0.5 * (sz1 + sz2)
    Jx = 0.5 * (sx1 + sx2)
    return a, b, p, sm1, sm2, Jz, Jx

a, b, p, sm1, sm2, Jz, Jx = build_operators()

sz1_reduced = tensor(sigmaz(), qeye(2))
sz2_reduced = tensor(qeye(2), sigmaz())
I_reduced = tensor(qeye(2), qeye(2))

# Local Battery Hamiltonian (Shifted ground state to zero energy)
H_B_qobj = (w_q / 2.0) * (sz1_reduced + I_reduced) + (w_q / 2.0) * (sz2_reduced + I_reduced)
H_B_evals = np.sort(H_B_qobj.eigenenergies())

c_ops = [
    np.sqrt(kappa_c) * a,
    np.sqrt(kappa_m) * b,
    np.sqrt(kappa_p) * p,
    np.sqrt(gamma_q) * sm1,
    np.sqrt(gamma_q) * sm2
]

# ==========================================
# 4. Core Simulation Engine (Ergotropy & Gap)
# ==========================================
def simulate_battery_dynamics(delta, g_b):
    H_CMM = (w_c + delta) * a.dag() * a + w_m * b.dag() * b + w_p * p.dag() * p \
            + g_a * (a + a.dag()) * (b + b.dag()) \
            + g_b * (b + b.dag()) * (p + p.dag())
            
    H_QB = w_q * Jz
    H_int = 2 * lam * (a + a.dag()) * Jx
    H_full = H_CMM + H_QB + H_int
    
    # Spectral Gap Calculation
    L = liouvillian(H_full, c_ops)
    try:
        evals, _ = eigs(L.data, k=15, which='LR', tol=1e-4)
        evals = np.real(evals)
    except:
        evals = np.real(L.eigenenergies()) 
        
    evals_rounded = np.round(evals, 5) 
    unique_evals = np.sort(np.unique(evals_rounded))[::-1] 
    
    non_zero = unique_evals[unique_evals < -1e-4]
    
    if len(non_zero) >= 2:
        spectral_gap = np.abs(non_zero[0] - non_zero[1])
    elif len(non_zero) == 1:
        spectral_gap = np.abs(non_zero[0])
    else:
        spectral_gap = 0.0
    
    # Ergotropy Dynamics
    psi_init = tensor(basis(Nc, 2), basis(Nm,0), basis(Np,0), basis(2,1), basis(2,1))
    rho_init = ket2dm(psi_init)
    
    res = mesolve(H_full, rho_init, tlist, c_ops, [])
    
    ergo_t = []
    eff_t = []
    
    for state in res.states:
        rho_b = state.ptrace([3, 4]) 
        energy = expect(H_B_qobj, rho_b)
        
        rho_b_evals = np.sort(np.real(rho_b.eigenenergies()))[::-1]
        E_passive = np.sum(rho_b_evals * H_B_evals)
        
        ergotropy = max(0.0, energy - E_passive)
        ergo_t.append(ergotropy)
        eff_t.append(ergotropy / energy if energy > 1e-6 else 0)
        
    return np.array(ergo_t), np.array(eff_t), spectral_gap

# ==========================================
# 5. Non-Markovianity (BLP Measure) function
# ==========================================
def calculate_blp_measure(delta, g_b):
    H_CMM = (w_c + delta)*a.dag()*a + w_m*b.dag()*b + w_p*p.dag()*p \
            + g_a*(a+a.dag())*(b+b.dag()) + g_b*(b+b.dag())*(p+p.dag())
    H_full = H_CMM + w_q*Jz + 2*lam*(a+a.dag())*Jx
    
    # Two orthogonal initial states for the battery qubits
    psi1 = tensor(basis(Nc, 2), basis(Nm,0), basis(Np,0), basis(2,1), basis(2,0))
    psi2 = tensor(basis(Nc, 2), basis(Nm,0), basis(Np,0), basis(2,0), basis(2,1))
    
    res1 = mesolve(H_full, ket2dm(psi1), tlist, c_ops, [])
    res2 = mesolve(H_full, ket2dm(psi2), tlist, c_ops, [])
    
    trace_dists = [tracedist(res1.states[i].ptrace([3,4]), res2.states[i].ptrace([3,4])) for i in range(len(tlist))]
    derivs = np.gradient(trace_dists, tlist[1]-tlist[0])
    return simpson(np.maximum(derivs, 0), x=tlist)

# ==========================================
# 6. Smart Execution & Checkpointing
# ==========================================
if __name__ == "__main__":
    print("Initiating Smart Checkpoint Simulation...")
    start_time = time.time()

    csv_file = "CMM_Battery_LEP_Results_Final.csv"
    file_exists = os.path.isfile(csv_file)

    computed_pairs = set()
    if file_exists:
        with open(csv_file, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                computed_pairs.add((float(row['Delta']), float(row['g_b'])))
        print(f"Found {len(computed_pairs)} previously computed points. Resuming...")

    with open(csv_file, mode='a', newline='') as file:
        fieldnames = ['Delta', 'g_b', 'Spectral_Gap', 'Max_Ergotropy', 'Max_Efficiency']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()

        print("Scanning Parameter Space...")
        for gb in gb_scan:
            for dlt in delta_scan:
                dlt_rnd = round(dlt, 3)
                gb_rnd = round(gb, 3)
                
                if (dlt_rnd, gb_rnd) in computed_pairs:
                    continue
                    
                print(f"Computing: Delta={dlt_rnd}, g_b={gb_rnd} ...")
                ergo, eff, gap = simulate_battery_dynamics(dlt, gb)
                
                writer.writerow({
                    'Delta': dlt_rnd, 'g_b': gb_rnd,
                    'Spectral_Gap': round(gap, 5), 'Max_Ergotropy': round(np.max(ergo), 4),
                    'Max_Efficiency': round(np.max(eff), 4)
                })
                file.flush() 
                os.fsync(file.fileno())

    print("Grid Data successfully completed and fully saved.")

    # ==========================================
    # 7. Plotting
    # ==========================================
    print("Generating Figures from saved data...")

    gap_matrix = np.zeros((len(gb_scan), len(delta_scan)))
    ergo_matrix = np.zeros((len(gb_scan), len(delta_scan)))

    with open(csv_file, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dlt_val = float(row['Delta'])
            gb_val = float(row['g_b'])
            
            i = (np.abs(gb_scan - gb_val)).argmin()
            j = (np.abs(delta_scan - dlt_val)).argmin()
            
            gap_matrix[i, j] = float(row['Spectral_Gap'])
            ergo_matrix[i, j] = float(row['Max_Ergotropy'])

    # --- Figure 1 ---
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    X, Y = np.meshgrid(delta_scan, gb_scan)

    cp1 = ax1.contourf(X, Y, gap_matrix, levels=40, cmap='viridis_r')
    fig1.colorbar(cp1, ax=ax1, label=r'Liouvillian Spectral Gap $\Delta\lambda$')
    ax1.set_title(r'(a) LEP Formation Landscape')
    ax1.set_xlabel(r'Detuning $\Delta$')
    ax1.set_ylabel(r'Magnon-Phonon Coupling $g_b$')

    cp2 = ax2.contourf(X, Y, ergo_matrix, levels=40, cmap='magma')
    fig1.colorbar(cp2, ax=ax2, label=r'Max Ergotropy $\mathcal{E}_{max}$')
    ax2.set_title(r'(b) Battery Charging Performance')
    ax2.set_xlabel(r'Detuning $\Delta$')
    ax2.set_ylabel(r'Magnon-Phonon Coupling $g_b$')
    plt.tight_layout()
    plt.savefig('Fig1_LEP_Ergotropy_Phase_Diagram.png')
    plt.close()

    # --- Figure 2 ---
    print("Computing BLP Measure ONLY for the optimal Delta slice...")
    optimal_delta_idx = np.unravel_index(np.argmax(ergo_matrix), ergo_matrix.shape)[1]
    opt_delta = delta_scan[optimal_delta_idx]
    ergo_slice = ergo_matrix[:, optimal_delta_idx]

    blp_slice = []
    for gb in gb_scan:
        blp = calculate_blp_measure(opt_delta, gb)
        blp_slice.append(blp)

    fig2, ax = plt.subplots(figsize=(8, 6))
    color = 'tab:red'
    ax.plot(gb_scan, ergo_slice, 'ro-', lw=2, label=r'Ergotropy $\mathcal{E}$')
    ax.set_xlabel(r'Phonon Coupling Strength $g_b$')
    ax.set_ylabel(r'Max Ergotropy $\mathcal{E}_{max}$', color=color)
    ax.tick_params(axis='y', labelcolor=color)

    ax2 = ax.twinx()
    color = 'tab:blue'
    ax2.plot(gb_scan, blp_slice, 'bs--', lw=2, label=r'BLP Measure $\mathcal{N}$')
    ax2.set_ylabel(r'Non-Markovianity $\mathcal{N}_{BLP}$', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    plt.title(r'Interplay of Ergotropy and Phonon-Induced Memory')
    fig2.tight_layout()
    plt.savefig('Fig2_NonMarkovian_Hypothesis.png')
    plt.close()

    # --- Figure 3 ---
    best_idx = np.unravel_index(np.argmax(ergo_matrix), ergo_matrix.shape)
    opt_gb = gb_scan[best_idx[0]]

    ergo_opt, eff_opt, _ = simulate_battery_dynamics(opt_delta, opt_gb)
    ergo_bad, eff_bad, _ = simulate_battery_dynamics(0.0, 0.0) 

    fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(tlist, ergo_opt, 'r-', lw=2, label=rf'Optimal LEP ($\Delta={opt_delta:.2f}, g_b={opt_gb:.2f}$)')
    ax1.plot(tlist, ergo_bad, 'k--', lw=2, label='Markovian Baseline')
    ax1.set_xlabel('Time $t$')
    ax1.set_ylabel(r'Ergotropy $\mathcal{E}(t)$')
    ax1.set_title('(a) Ergotropy Protection Dynamics')
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(tlist, eff_opt, 'b-', lw=2, label='Optimal LEP')
    ax2.plot(tlist, eff_bad, 'k--', lw=2, label='Baseline')
    ax2.set_xlabel('Time $t$')
    ax2.set_ylabel(r'Quantum Efficiency $\eta_Q(t)$')
    ax2.set_title('(b) Charging Efficiency')
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('Fig3_Optimal_Dynamics.png')
    plt.close()

    print(f"Simulation Complete! Total Time: {(time.time() - start_time):.2f} seconds.")

