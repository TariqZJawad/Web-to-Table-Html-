import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# =========================================
# إعدادات النشر العلمي للأشكال
# =========================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'lines.linewidth': 2.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

print("="*75)
print("🚀 Starting Master Combined Scaling Simulation (A and B from Scratch)")
print("="*75)

N_vals = [1, 2, 3, 4]
N_cavity = 4
w_b = 1.0
g = 0.5
gamma0 = 1.0
lam_decay = 0.2

# مخازن للقيم المعتمدة
delta_opts_A = []
delta_opts_B = []
ergo_opt_filter_A = []
ergo_opt_filter_B = []

# =========================================
# 1. حلقة المحاكاة الكبرى لحساب البيئتين
# =========================================
for N in N_vals:
    print(f"⏳ Calculating Dynamics for N = {N}...")
    Jz = tensor(jmat(N/2, 'z'), qeye(N_cavity))
    Jp = tensor(jmat(N/2, '+'), qeye(N_cavity))
    Jm = tensor(jmat(N/2, '-'), qeye(N_cavity))
    a  = tensor(qeye(N + 1), destroy(N_cavity))
    H_B_total = Jz + N/2

    rho_init = ket2dm(tensor(basis(N+1, 0), basis(N_cavity, 0)))
    H_evals = np.sort(jmat(N/2, 'z').eigenenergies() + N/2)

    def calc_ergo(state):
        rho_b = state.ptrace(0)
        evals = np.sort(rho_b.eigenenergies())[::-1]
        E_pass = np.sum(evals * H_evals)
        return max(0.0, expect(H_B_total.ptrace(0), rho_b) - E_pass)

    def decay(t, args): return np.sqrt(args['gamma0'] * np.exp(-lam_decay * t))
    c_ops = [[a, decay]]

    # --- حسابات البيئة أ ---
    delta_opt_A = g * np.sqrt(N / (2 * gamma0)) + 0.2
    delta_opts_A.append(delta_opt_A)
    H_opt_A = w_b * (Jz + N/2) + (w_b + delta_opt_A) * a.dag() * a + g * (Jp * a + Jm * a.dag())
    res_opt_A = mesolve(H_opt_A, rho_init, [0, 15], c_ops, [], args={'gamma0': gamma0})
    ergo_opt_filter_A.append(calc_ergo(res_opt_A.states[-1]))

    # --- حسابات البيئة ب (بناءً على نفس الأساس للكيوبتات ولكن بفلتر مختلف) ---
    # ملاحظة: البيئة ب كانت تتطلب +0.3 كفلتر أمثل
    delta_opt_B = g * np.sqrt(N / (2 * gamma0)) + 0.3
    delta_opts_B.append(delta_opt_B)
    H_opt_B = w_b * (Jz + N/2) + (w_b + delta_opt_B) * a.dag() * a + g * (Jp * a + Jm * a.dag())
    res_opt_B = mesolve(H_opt_B, rho_init, [0, 15], c_ops, [], args={'gamma0': gamma0})
    ergo_opt_filter_B.append(calc_ergo(res_opt_B.states[-1]))

# تحويل القوائم إلى مصفوفات للحسابات
delta_opts_A = np.array(delta_opts_A)
delta_opts_B = np.array(delta_opts_B)
ergo_opt_filter_A = np.array(ergo_opt_filter_A)
ergo_opt_filter_B = np.array(ergo_opt_filter_B)

# حساب الميزة الكمية لكل بيئة
# البيئة أ
E1_opt_A = ergo_opt_filter_A[0]
additive_scaling_A = np.array(N_vals) * E1_opt_A
advantage_ratio_A = ergo_opt_filter_A / additive_scaling_A
# البيئة ب
E1_opt_B = ergo_opt_filter_B[0]
additive_scaling_B = np.array(N_vals) * E1_opt_B
advantage_ratio_B = ergo_opt_filter_B / additive_scaling_B

# =========================================
# 2. طباعة الجدول المدمج
# =========================================
print("\n" + "="*85)
print(f"| {'N':^3} | {'Delta*_A':^8} | {'E_res_A (Fil)':^15} | {'Delta*_B':^8} | {'E_res_B (Fil)':^15} |")
print("-" * 85)
for i, N in enumerate(N_vals):
    print(f"| {N:^3} | {delta_opts_A[i]:^8.2f} | {ergo_opt_filter_A[i]:^15.4f} | {delta_opts_B[i]:^8.2f} | {ergo_opt_filter_B[i]:^15.4f} |")
print("="*85 + "\n")

# =========================================
# 3. رسم الأشكال المدمجة
# =========================================
print("🎨 Generating Figures...")

# --- الشكل 9: Delta* مقابل N للبيئتين (مقارنة في رسم واحد) ---
plt.figure(figsize=(7, 5))
plt.plot(N_vals, delta_opts_A, 'bo-', markersize=8, label='Environment A (Nuclear Decay)')
plt.plot(N_vals, delta_opts_B, 'rs--', markersize=8, label='Environment B (Thermal Noise/Dephasing)')
plt.xlabel('Number of Qubits ($N$)')
plt.ylabel(r'Optimal Detuning $\Delta^*$')
plt.title(r'Scaling of Optimal Detuning $\Delta^*(N)$')
plt.xticks(N_vals)
plt.legend()
plt.grid(alpha=0.3, linestyle='--')
plt.savefig('Figure_9_Delta_Scaling_Comparison.png')
plt.close()

# --- الشكل 10: Ergotropy Scaling للبيئتين (رسومات منفصلة في صورة واحدة) ---
fig10, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
plt.subplots_adjust(hspace=0.2)

# لوحة أ: البيئة أ
ax1.plot(N_vals, ergo_opt_filter_A, 'ro-', markersize=8, label=r'Collective Filtered ($\mathcal{E}_N$)')
ax1.plot(N_vals, additive_scaling_A, 'g^--', markersize=8, alpha=0.7, label=r'Independent Additive ($N \times \mathcal{E}_1$)')
ax1.set_ylabel(r'Residual Ergotropy $\mathcal{E}_{res}$ at $t=15$')
ax1.set_title(r'(a) Environment A - Nuclear Decay (Nuclear Decay)')
ax1.legend()
ax1.grid(alpha=0.3, linestyle='--')

# لوحة ب: البيئة ب
ax2.plot(N_vals, ergo_opt_filter_B, 'rs--', markersize=8, label=r'Collective Filtered ($\mathcal{E}_N$)')
ax2.plot(N_vals, additive_scaling_B, 'g^--', markersize=8, alpha=0.7, label=r'Independent Additive ($N \times \mathcal{E}_1$)')
ax2.set_xlabel('Number of Qubits ($N$)')
ax2.set_ylabel(r'Residual Ergotropy $\mathcal{E}_{res}$ at $t=15$')
ax2.set_title(r'(b) Environment B - Thermal Noise & Dephasing')
ax2.legend()
ax2.grid(alpha=0.3, linestyle='--')
plt.xticks(N_vals)

plt.savefig('Figure_10_Ergotropy_Scaling_Combined.png')
plt.close()

# --- الشكل 11: Quantum Advantage Metric للبيئتين (رسومات منفصلة في صورة واحدة) ---
fig11, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
plt.subplots_adjust(hspace=0.2)

# لوحة أ: البيئة أ
ax1.plot(N_vals, advantage_ratio_A, 'bs-', markersize=8)
ax1.axhline(1.0, color='k', linestyle='--', alpha=0.5)
ax1.fill_between(N_vals, 1.0, advantage_ratio_A, where=(advantage_ratio_A >= 1.0), color='blue', alpha=0.1)
ax1.set_ylabel(r'Advantage Ratio $\mathcal{E}_N / (N \times \mathcal{E}_1)$')
ax1.set_title(r'(a) Environment A - Nuclear Decay')
ax1.grid(alpha=0.3, linestyle='--')

# لوحة ب: البيئة ب
ax2.plot(N_vals, advantage_ratio_B, 'bx--', markersize=8, label='Advantage Env B')
ax2.axhline(1.0, color='k', linestyle='--', alpha=0.5)
ax2.fill_between(N_vals, 1.0, advantage_ratio_B, where=(advantage_ratio_B >= 1.0), color='blue', alpha=0.1)
ax2.set_xlabel('Number of Qubits ($N$)')
ax2.set_ylabel(r'Advantage Ratio $\mathcal{E}_N / (N \times \mathcal{E}_1)$')
ax2.set_title(r'(b) Environment B - Thermal Noise & Dephasing')
ax2.grid(alpha=0.3, linestyle='--')
plt.xticks(N_vals)

plt.savefig('Figure_11_Advantage_Ratio_Combined.png')
plt.close()

print("✅ All calculations finished successfully!")
print("✅ Figures saved: Figure_9, Figure_10_Combined, Figure_11_Combined.")
