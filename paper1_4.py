import numpy as np
import matplotlib.pyplot as plt
from qutip import *

print("🚀 Running N-Scaling Analysis (N=1 to 4) for Quantum Advantage...")

# ==========================================
# 1. إعدادات المحاكاة
# ==========================================
N_vals = [1, 2, 3, 4]
N_cavity = 4
w_b = 1.0
g = 0.5
lam_decay = 0.2
gamma0_fixed = 0.8  # نختار شدة اضمحلال متوسطة لنرى التأثير بوضوح
t_final = 15.0
tlist = np.linspace(0, t_final, 50)

# مصفوفات لتخزين النتائج
ergo_no_filter = []
ergo_opt_filter = []

# ==========================================
# 2. حلقة حساب الإرجوتروبي لكل N
# ==========================================
for N in N_vals:
    # إعداد المؤثرات
    Jz = tensor(jmat(N/2, 'z'), qeye(N_cavity))
    Jp = tensor(jmat(N/2, '+'), qeye(N_cavity))
    Jm = tensor(jmat(N/2, '-'), qeye(N_cavity))
    a  = tensor(qeye(N + 1), destroy(N_cavity))
    H_B_total = Jz + N/2

    # الحالة الابتدائية
    psi_init = tensor(basis(N+1, 0), basis(N_cavity, 0))
    rho_init = ket2dm(psi_init)

    # دالة الإرجوتروبي
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

    # 1. بدون فلتر (الرنين)
    H_0 = w_b * (Jz + N/2) + w_b * a.dag() * a + g * (Jp * a + Jm * a.dag())
    res_0 = mesolve(H_0, rho_init, tlist, c_ops, [], args=args)
    ergo_no_filter.append(calc_ergo(res_0.states[-1]))

    # 2. مع فلتر الإزاحة (نستخدم إزاحة متكيفة نظرياً لكل N لضمان العدل)
    delta_opt = g * np.sqrt(N / (2 * gamma0_fixed)) + 0.2 # إضافة 0.2 كتعويض للانزياح الذي اكتشفته!
    w_p = w_b + delta_opt
    H_opt = w_b * (Jz + N/2) + w_p * a.dag() * a + g * (Jp * a + Jm * a.dag())
    res_opt = mesolve(H_opt, rho_init, tlist, c_ops, [], args=args)
    ergo_opt_filter.append(calc_ergo(res_opt.states[-1]))

# ==========================================
# 3. حساب نسب الميزة الكمية (Quantum Advantage)
# ==========================================
ergo_no_filter = np.array(ergo_no_filter)
ergo_opt_filter = np.array(ergo_opt_filter)

# المقياس الخطي (Additive Scaling): ماذا لو كانت البطاريات تعمل بشكل منفصل تماماً؟
additive_scaling = np.array(N_vals) * ergo_opt_filter[0]

# النسبة التي طلبتها: الإرجوتروبي الفعلي / (N * إرجوتروبي بطارية واحدة)
advantage_ratio = ergo_opt_filter / additive_scaling

# ==========================================
# 4. الرسوم البيانية (The Scaling Figures)
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(r'Scaling of Residual Ergotropy with Qubit Number ($N$)', fontsize=16)

# الشكل الأول: التوسع الفائق
ax1.plot(N_vals, ergo_no_filter, 'ko--', markersize=8, label=r'No Filter ($\Delta=0$)')
ax1.plot(N_vals, additive_scaling, 'g^--', markersize=8, alpha=0.6, label='Independent Qubits ($N \\times \\epsilon_1$)')
ax1.plot(N_vals, ergo_opt_filter, 'ro-', markersize=8, linewidth=2, label='Optimal Filter (Collective)')

ax1.set_xticks(N_vals)
ax1.set_xlabel('Number of Qubits ($N$)', fontsize=12)
ax1.set_ylabel(r'Residual Ergotropy ($\epsilon_{res}$) at $t=15$', fontsize=12)
ax1.set_title('Superextensive vs Additive Scaling', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# الشكل الثاني: نسبة التحسين (Advantage Ratio)
# خط أفقي عند 1 (يفصل بين الميزة الكلاسيكية والكمية)
ax2.axhline(1.0, color='k', linestyle='--', alpha=0.5)
ax2.plot(N_vals, advantage_ratio, 'bs-', markersize=8, linewidth=2)

# تلوين المنطقة فوق 1 للإشارة إلى الميزة الكمية
ax2.fill_between(N_vals, 1.0, advantage_ratio, where=(advantage_ratio >= 1.0), color='blue', alpha=0.1)

ax2.set_xticks(N_vals)
ax2.set_xlabel('Number of Qubits ($N$)', fontsize=12)
ax2.set_ylabel(r'Advantage Ratio $\epsilon_N / (N \times \epsilon_1)$', fontsize=12)
ax2.set_title('Quantum Advantage Metric', fontsize=14)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("✅ Analysis Complete! The scaling charts are ready.")
