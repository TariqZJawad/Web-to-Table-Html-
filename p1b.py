import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from scipy.integrate import simpson
import csv

# ==========================================
# إعدادات النشر العلمي (Serif Fonts & High DPI)
# ==========================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'lines.linewidth': 2.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

def run_envB_reproducible_protocol(N_qubits):
    print(f"\n{'='*65}")
    print(f"🔥 Analyzing Environment B (Thermal & Noise) for N = {N_qubits}")
    print(f"{'='*65}")

    # المعاملات الفيزيائية
    N_cavity = 4
    w_b = 1.0
    g = 0.5
    lam_decay = 0.2
    gamma_phi = 0.05    # قوة فك الترابط (Dephasing)
    n0_fixed = 0.5      # تثبيت سعة الضجيج الحراري لرسم الخريطة مقابل gamma0
    W_th = 1.0          # تردد الدورات الحرارية

    # المؤثرات
    Jz = tensor(jmat(N_qubits/2, 'z'), qeye(N_cavity))
    Jp = tensor(jmat(N_qubits/2, '+'), qeye(N_cavity))
    Jm = tensor(jmat(N_qubits/2, '-'), qeye(N_cavity))
    a  = tensor(qeye(N_qubits + 1), destroy(N_cavity))
    H_B_total = Jz + N_qubits/2

    # الحالات الابتدائية
    rho1_init = ket2dm(tensor(basis(N_qubits+1, 0), basis(N_cavity, 0)))
    rho2_init = ket2dm(tensor(basis(N_qubits+1, N_qubits), basis(N_cavity, 0)))

    H_evals = np.sort(jmat(N_qubits/2, 'z').eigenenergies() + N_qubits/2)
    def calc_ergo(state):
        rho_b = state.ptrace(0)
        evals = np.sort(rho_b.eigenenergies())[::-1]
        E_pass = np.sum(evals * H_evals)
        return max(0.0, expect(H_B_total.ptrace(0), rho_b) - E_pass)

    def c_op_emit(t, args):
        nth = args['n0'] * (1 + np.sin(args['W'] * t)**2)
        return np.sqrt(args['gamma0'] * (nth + 1))

    def c_op_absorb(t, args):
        nth = args['n0'] * (1 + np.sin(args['W'] * t)**2)
        return np.sqrt(args['gamma0'] * nth)

    # ---------------------------------------------------------
    # 1. الديناميكا الزمنية (للحصول على أرقام الجدول)
    # ---------------------------------------------------------
    tlist_dyn = np.linspace(0, 15, 150)
    def run_dyn(delta, gamma0):
        H = w_b * (Jz + N_qubits/2) + (w_b + delta) * a.dag() * a + g * (Jp * a + Jm * a.dag())
        args = {'gamma0': gamma0, 'n0': n0_fixed, 'W': W_th}
        c_ops = [[a, c_op_emit], [a.dag(), c_op_absorb], np.sqrt(gamma_phi) * Jz]
        
        res1 = mesolve(H, rho1_init, tlist_dyn, c_ops, [], args=args)
        res2 = mesolve(H, rho2_init, tlist_dyn, c_ops, [], args=args)
        
        ergo = [calc_ergo(s) for s in res1.states]
        dist = [tracedist(res1.states[i].ptrace(0), res2.states[i].ptrace(0)) for i in range(len(tlist_dyn))]
        blp = simpson(np.maximum(np.gradient(dist, tlist_dyn[1]-tlist_dyn[0]), 0), x=tlist_dyn)
        return ergo, dist, blp

    print("Step 1: Calculating Time Dynamics...")
    # حساب الإزاحة المثلى نظرياً بناءً على النسبة الجماعية
    delta_opt_val = g * np.sqrt(N_qubits / (2 * 1.0)) + 0.3 # عند gamma0 = 1.0
    
    ergo_0, dist_0, blp_0 = run_dyn(0.0, 1.0)
    ergo_opt, dist_opt, blp_opt = run_dyn(delta_opt_val, 1.0)

    # الرسم البياني للديناميكا
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9))
    ax1.plot(tlist_dyn, dist_0, 'k--', label=fr'$\Delta=0, \mathcal{{N}}={blp_0:.3f}$')
    ax1.plot(tlist_dyn, dist_opt, 'b-', label=fr'$\Delta={delta_opt_val:.2f}, \mathcal{{N}}={blp_opt:.3f}$')
    ax1.set_ylabel(r'$D(\rho_1, \rho_2)$')
    ax1.legend(loc='upper right'); ax1.grid(alpha=0.2)
    
    ax2.plot(tlist_dyn, ergo_0, 'k--', label='Unfiltered')
    ax2.plot(tlist_dyn, ergo_opt, 'r-', label='Filtered')
    ax2.set_xlabel(r'Time $t$'); ax2.set_ylabel(r'Ergotropy $\mathcal{E}(t)$')
    ax2.legend(loc='upper right'); ax2.grid(alpha=0.2)
    plt.savefig(f'Dynamics_EnvB_N{N_qubits}.png')
    plt.close()

    # ---------------------------------------------------------
    # 2. الخريطة الحرارية (X=Delta, Y=Gamma0) لإرجاع المنحنى
    # ---------------------------------------------------------
    print("Step 2: Generating Curved Heatmap (Delta vs Gamma0)...")
    delta_vals = np.linspace(0.0, 2.5, 20)
    gamma_vals = np.linspace(0.1, 1.5, 20)
    ergo_map = np.zeros((len(gamma_vals), len(delta_vals)))

    for i, g0 in enumerate(gamma_vals):
        args_hm = {'gamma0': g0, 'n0': n0_fixed, 'W': W_th}
        c_ops_hm = [[a, c_op_emit], [a.dag(), c_op_absorb], np.sqrt(gamma_phi) * Jz]
        for j, delta in enumerate(delta_vals):
            H_hm = w_b * (Jz + N_qubits/2) + (w_b + delta) * a.dag() * a + g * (Jp * a + Jm * a.dag())
            res = mesolve(H_hm, rho1_init, [0, 15], c_ops_hm, [], args=args_hm)
            ergo_map[i, j] = calc_ergo(res.states[-1])

    plt.figure(figsize=(8, 6))
    X, Y = np.meshgrid(delta_vals, gamma_vals)
    plt.contourf(X, Y, ergo_map, levels=40, cmap='viridis')
    plt.colorbar(label=r'$\mathcal{E}_{res}$ at $t=15$')
    
    # رسم المنحنى النظري (الذي كان منحني في صورك القديمة)
    theo_opt = g * np.sqrt(N_qubits / (2 * gamma_vals)) + 0.3
    plt.plot(theo_opt, gamma_vals, 'w--', linewidth=2, label=r'Theoretical $\Delta^*$')
    
    plt.title(f'Env B Survival Map ($N={N_qubits}$)')
    plt.xlabel(r'Detuning $\Delta$'); plt.ylabel(r'Decay Rate $\gamma_0$')
    plt.xlim(0, 2.5); plt.ylim(0.1, 1.5); plt.legend(loc='lower right')
    plt.savefig(f'Survival_Map_EnvB_N{N_qubits}.png')
    plt.close()

    return {
        'N': N_qubits, 'Delta_opt': round(delta_opt_val, 2),
        'E0': round(ergo_0[-1], 4), 'Eopt': round(ergo_opt[-1], 4),
        'BLP_0': round(blp_0, 4), 'BLP_opt': round(blp_opt, 4)
    }

# تشغيل واستخراج الجدول
results = [run_envB_reproducible_protocol(N) for N in [2, 3, 4]]

# طباعة الجدول
print("\n" + "="*85)
print(f"| {'N':^3} | {'Delta*':^8} | {'E0 (No Filter)':^15} | {'Eopt (Filtered)':^15} | {'BLP (Unfil)':^12} | {'BLP (Fil)':^10} |")
print("-" * 85)
for r in results:
    print(f"| {r['N']:^3} | {r['Delta_opt']:^8.2f} | {r['E0']:^15.4f} | {r['Eopt']:^15.4f} | {r['BLP_0']:^12.4f} | {r['BLP_opt']:^10.4f} |")
print("="*85)

# حفظ الجدول
with open("Final_Results_EnvB.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader(); writer.writerows(results)
