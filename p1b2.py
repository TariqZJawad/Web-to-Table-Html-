import numpy as np
import matplotlib.pyplot as plt
from qutip import *

print("🚀 Starting Corrected N-Scaling Analysis...")

# ==========================================
# 1. إعدادات النظام
# ==========================================
N_vals = [1, 2, 3, 4] # جرب 4 أولاً للتأكد من استقرار الذاكرة
N_cavity = 10         # رفعنا القيمة لضمان دقة النتائج ومنع الاقتطاع
w_b = 1.0
g = 0.5
lam_decay = 0.2
gamma0_fixed = 0.8
tlist = np.linspace(0, 15.0, 50)

ergo_no_filter_avg = []
ergo_opt_filter_avg = []

for N in N_vals:
    print(f"🔄 Processing N = {N}...")
    
    # تعريف المؤثرات مع التأكد من أنها Qobj
    # ملاحظة: jmat تعيد Qobj بالفعل، لكن tensor تدمجها
    iden_b = qeye(N + 1)
    iden_c = qeye(N_cavity)
    
    Jz = tensor(jmat(N/2, 'z'), iden_c)
    Jp = tensor(jmat(N/2, '+'), iden_c)
    Jm = tensor(jmat(N/2, '-'), iden_c)
    a  = tensor(iden_b, destroy(N_cavity))
    
    # الإصلاح الأهم: تحويل الرقم (N/2) إلى مؤثر هوية بنفس الأبعاد
    H_B_operator = Jz + (N/2) * tensor(iden_b, iden_c)

    # الحالة الابتدائية
    psi_init = tensor(basis(N+1, 0), basis(N_cavity, 0))
    rho_init = ket2dm(psi_init)

    # حساب القيم الذاتية للهاميلتونيان الأساسي (Passive Energy)
    # نأخذ قيم الكيوبت فقط ptrace(0)
    H_b_only = jmat(N/2, 'z') + (N/2) * iden_b
    H_evals = np.sort(H_b_only.eigenenergies())

    def calc_ergo(state):
        # التأكد من أن الحالة Qobj قبل ptrace
        rho_b = state.ptrace(0)
        evals = np.sort(rho_b.eigenenergies())[::-1]
        E_curr = expect(H_b_only, rho_b)
        E_pass = np.sum(evals * H_evals)
        return max(0, E_curr - E_pass)

    def decay(t, args): 
        return np.sqrt(args['gamma0'] * np.exp(-lam_decay * t))
    
    c_ops = [[a, decay]]
    args = {'gamma0': gamma0_fixed}

    # 1. بدون فلتر (الرنين)
    H_0 = w_b * Jz + (N/2)*w_b*tensor(iden_b, iden_c) + w_b * a.dag() * a + g * (Jp * a + Jm * a.dag())
    res_0 = mesolve(H_0, rho_init, tlist, c_ops, [], args=args)
    ergo_traj_0 = [calc_ergo(s) for s in res_0.states]
    ergo_no_filter_avg.append(np.mean(ergo_traj_0))

    # 2. الفلتر المثالي
    delta_opt = g * np.sqrt(N / (2 * gamma0_fixed)) + 0.2
    w_p = w_b + delta_opt
    H_opt = w_b * Jz + (N/2)*w_b*tensor(iden_b, iden_c) + w_p * a.dag() * a + g * (Jp * a + Jm * a.dag())
    res_opt = mesolve(H_opt, rho_init, tlist, c_ops, [], args=args)
    ergo_traj_opt = [calc_ergo(s) for s in res_opt.states]
    ergo_opt_filter_avg.append(np.mean(ergo_traj_opt))

# تحويل النتائج لمصفوفات NumPy للحساب الرياضي
ergo_no_filter_avg = np.array(ergo_no_filter_avg)
ergo_opt_filter_avg = np.array(ergo_opt_filter_avg)

# حساب الإضافية (Additive Scaling)
# هنا نحول N_vals لمصفوفة لنتجنب TypeError السابق
additive_scaling = np.array(N_vals) * ergo_opt_filter_avg[0]
advantage_ratio = ergo_opt_filter_avg / additive_scaling

# ==========================================
# الرسم البياني
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(N_vals, ergo_no_filter_avg, 'ko--', label='No Filter')
ax1.plot(N_vals, additive_scaling, 'g^--', label='Independent (N*E1)')
ax1.plot(N_vals, ergo_opt_filter_avg, 'ro-', label='Optimal Filter')
ax1.set_xlabel('N Qubits')
ax1.set_ylabel('Avg Ergotropy')
ax1.legend()

ax2.plot(N_vals, advantage_ratio, 'bs-')
ax2.axhline(1.0, color='k', linestyle='--')
ax2.set_xlabel('N Qubits')
ax2.set_ylabel('Advantage Ratio')

plt.tight_layout()
plt.show()

print("✅ Success! Simulation completed without TypeErrors.")
