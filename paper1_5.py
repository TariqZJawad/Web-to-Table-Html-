import numpy as np
import matplotlib.pyplot as plt
from qutip import *

print("🚀 Running N-Scaling Analysis (N=1 to 5) using TIME-AVERAGED Ergotropy...")

N_vals = [1, 2, 3, 4, 5]
N_cavity = 6  # زدنا الحجم قليلاً ليتحمل N=5
w_b = 1.0
g = 0.5
lam_decay = 0.2
gamma0_fixed = 0.8
tlist = np.linspace(0, 15.0, 50)

ergo_no_filter_avg = []
ergo_opt_filter_avg = []

for N in N_vals:
    Jz = tensor(jmat(N/2, 'z'), qeye(N_cavity))
    Jp = tensor(jmat(N/2, '+'), qeye(N_cavity))
    Jm = tensor(jmat(N/2, '-'), qeye(N_cavity))
    a  = tensor(qeye(N + 1), destroy(N_cavity))
    H_B_total = Jz + N/2

    psi_init = tensor(basis(N+1, 0), basis(N_cavity, 0))
    rho_init = ket2dm(psi_init)

    H_evals = np.sort(jmat(N/2, 'z').eigenenergies() + N/2)
    def calc_ergo(state):
        rho_b = state.ptrace(0)
        evals = np.sort(rho_b.eigenenergies())[::-1]
        E_curr = expect(H_B_total.ptrace(0), rho_b)
        E_pass = np.sum(evals * H_evals)
        return max(0, E_curr - E_pass)

    def decay(t, args): return np.sqrt(args['gamma0'] * np.exp(-lam_decay * t))
    c_ops = [[a, decay]]
    args = {'gamma0': gamma0_fixed}

    print(f"Simulating N={N}...")
    
    # بدون فلتر
    H_0 = w_b * (Jz + N/2) + w_b * a.dag() * a + g * (Jp * a + Jm * a.dag())
    res_0 = mesolve(H_0, rho_init, tlist, c_ops, [], args=args)
    ergo_traj_0 = [calc_ergo(s) for s in res_0.states]
    ergo_no_filter_avg.append(np.mean(ergo_traj_0)) # نأخذ المتوسط الزمني

    # مع الفلتر المثالي الخاص بـ N
    delta_opt = g * np.sqrt(N / (2 * gamma0_fixed)) + 0.2
    w_p = w_b + delta_opt
    H_opt = w_b * (Jz + N/2) + w_p * a.dag() * a + g * (Jp * a + Jm * a.dag())
    res_opt = mesolve(H_opt, rho_init, tlist, c_ops, [], args=args)
    ergo_traj_opt = [calc_ergo(s) for s in res_opt.states]
    ergo_opt_filter_avg.append(np.mean(ergo_traj_opt)) # نأخذ المتوسط الزمني

# تحويل القوائم إلى مصفوفات
ergo_no_filter_avg = np.array(ergo_no_filter_avg)
ergo_opt_filter_avg = np.array(ergo_opt_filter_avg)

# حساب خط الأساس المستقل (N * E_1)
additive_scaling = np.array(N_vals) * ergo_opt_filter_avg[0]

# مقياس الميزة الكمية
advantage_ratio = ergo_opt_filter_avg / additive_scaling

# ==========================================
# الرسم البياني
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(r'Time-Averaged Residual Ergotropy Scaling ($N=1$ to $5$)', fontsize=16)

# الشكل الأيسر
ax1.plot(N_vals, ergo_no_filter_avg, 'ko--', markersize=8, label=r'No Filter ($\Delta=0$)')
ax1.plot(N_vals, additive_scaling, 'g^--', markersize=8, alpha=0.6, label=r'Independent Qubits ($N \times \langle\epsilon_1\rangle$)')
ax1.plot(N_vals, ergo_opt_filter_avg, 'ro-', markersize=8, linewidth=2, label='Optimal Filter (Collective)')

ax1.set_xticks(N_vals)
ax1.set_xlabel('Number of Qubits ($N$)', fontsize=12)
ax1.set_ylabel(r'Time-Averaged Ergotropy $\langle \epsilon \rangle$', fontsize=12)
ax1.set_title('Superextensive vs Additive Scaling', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# الشكل الأيمن
ax2.axhline(1.0, color='k', linestyle='--', alpha=0.5)
ax2.plot(N_vals, advantage_ratio, 'bs-', markersize=8, linewidth=2)
ax2.fill_between(N_vals, 1.0, advantage_ratio, where=(advantage_ratio >= 1.0), color='blue', alpha=0.1)

ax2.set_xticks(N_vals)
ax2.set_xlabel('Number of Qubits ($N$)', fontsize=12)
ax2.set_ylabel(r'Advantage Ratio $\langle\epsilon_N\rangle / (N \times \langle\epsilon_1\rangle)$', fontsize=12)
ax2.set_title('Quantum Advantage Metric (Time-Averaged)', fontsize=14)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n--- FINAL RESULTS ---")
for i, n in enumerate(N_vals):
    print(f"N={n} | Advantage Ratio = {advantage_ratio[i]:.3f}")
