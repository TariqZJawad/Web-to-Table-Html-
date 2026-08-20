import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# ==========================================
# 1. إعدادات النظام الجماعي (N-Qubits)
# ==========================================
N_qubits = 2          # نبدأ بـ 2 لتسريع الحسابات
N_cavity = 4
w_b = 1.0
g = 0.5
lam_decay = 0.2
t_final = 15.0        # الزمن الذي نقيس عنده "بقاء" الإرجوتروبي
tlist = np.linspace(0, t_final, 50)

# المؤثرات الجماعية
Jz = tensor(jmat(N_qubits/2, 'z'), qeye(N_cavity))
Jp = tensor(jmat(N_qubits/2, '+'), qeye(N_cavity))
Jm = tensor(jmat(N_qubits/2, '-'), qeye(N_cavity))
a  = tensor(qeye(N_qubits + 1), destroy(N_cavity)) # البعد المصحح N+1
H_B_total = Jz + N_qubits/2

# الحالة الابتدائية: البطاريات مشحونة بالكامل
psi_init = tensor(basis(N_qubits+1, 0), basis(N_cavity, 0))
rho_init = ket2dm(psi_init)

# دالة الإرجوتروبي (تم تحسينها لتكون أسرع في الحلقات التكرارية)
H_evals = np.sort(jmat(N_qubits/2, 'z').eigenenergies() + N_qubits/2)
def calc_final_ergotropy(state):
    rho_b = state.ptrace(0)
    evals = np.sort(rho_b.eigenenergies())[::-1]
    E_current = expect(H_B_total.ptrace(0), rho_b)
    E_passive = np.sum(evals * H_evals)
    return max(0, E_current - E_passive)

# ==========================================
# 2. إعداد شبكة المعاملات (Parameter Grid)
# ==========================================
# سنمسح 20 قيمة للإزاحة و 20 قيمة لشدة الاضمحلال (المجموع 400 محاكاة)
delta_vals = np.linspace(0.0, 2.0, 20)
gamma_vals = np.linspace(0.1, 1.5, 20)

# مصفوفة لتخزين النتائج
ergo_map = np.zeros((len(gamma_vals), len(delta_vals)))

print("Starting Heatmap Computations... (This will run 400 fast simulations)")

# ==========================================
# 3. حلقة المسح الشامل (Parameter Sweep)
# ==========================================
for i, gamma0 in enumerate(gamma_vals):
    # دالة الاضمحلال لهذه الدورة
    def nuclear_decay(t, args):
        return np.sqrt(args['gamma0'] * np.exp(-lam_decay * t))
    
    c_ops = [[a, nuclear_decay]]
    args = {'gamma0': gamma0}
    
    for j, delta in enumerate(delta_vals):
        w_p = w_b + delta
        H = w_b * (Jz + N_qubits/2) + w_p * a.dag() * a + g * (Jp * a + Jm * a.dag())
        
        # تشغيل المحاكاة
        res = mesolve(H, rho_init, tlist, c_ops, [], args=args)
        
        # حساب الإرجوتروبي المتبقي عند اللحظة الأخيرة
        final_state = res.states[-1]
        ergo_map[i, j] = calc_final_ergotropy(final_state)
    
    # طباعة التقدم حتى لا تظن أن الحاسوب تجمد
    if (i+1) % 5 == 0:
        print(f"Progress: {i+1}/20 Gamma values calculated...")

# ==========================================
# 4. رسم الخريطة الحرارية (The Money Shot)
# ==========================================
plt.figure(figsize=(10, 7))

# استخدام contourf لرسم خريطة ناعمة وملونة
X, Y = np.meshgrid(delta_vals, gamma_vals)
cp = plt.contourf(X, Y, ergo_map, levels=30, cmap='magma')
plt.colorbar(cp, label='Residual Ergotropy at $t=15$')

# رسم خط نظري متوقع للإزاحة المثلى (اختياري للزينة الأكاديمية)
# بناءً على معادلتك: Delta* ~ g * sqrt(N / 2*gamma)
theoretical_opt = g * np.sqrt(N_qubits / (2 * gamma_vals))
plt.plot(theoretical_opt, gamma_vals, 'w--', linewidth=2, alpha=0.8, label='Theoretical $\Delta^*$')

plt.title('Ergotropy Survival Map (N=2 Qubits in Nuclear Decay)', fontsize=14)
plt.xlabel(r'Detuning Filter ($\Delta$)', fontsize=12)
plt.ylabel(r'Initial Decay Strength ($\gamma_0$)', fontsize=12)
plt.legend(loc='upper right')
plt.xlim(0, 2.0)
plt.ylim(0.1, 1.5)
plt.tight_layout()
plt.show()
