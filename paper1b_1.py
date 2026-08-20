import numpy as np
import matplotlib.pyplot as plt
from qutip import *

print("🚀 Starting Realistic Lab Simulation: Thermal Cycles (Day/Night)...")

# ==========================================
# 1. إعدادات المصفوفة الجماعية
# ==========================================
N_qubits = 2       # نبدأ بـ 2 لتوضيح الفكرة بسرعة
N_cavity = 5
w_b = 1.0
g = 0.5
gamma_th = 0.15    # قوة الارتباط بالخزان الحراري

# معاملات الدورة الحرارية المتغيرة زمنياً
T_avg = 0.6        # متوسط درجة الحرارة
T_amp = 0.4        # سعة التذبذب (الفرق بين أقصى وأدنى حرارة)
tau_cycle = 10.0   # زمن الدورة الواحدة

tlist = np.linspace(0, 25.0, 250)

# ==========================================
# 2. المؤثرات والحالة الابتدائية
# ==========================================
Jz = tensor(jmat(N_qubits/2, 'z'), qeye(N_cavity))
Jp = tensor(jmat(N_qubits/2, '+'), qeye(N_cavity))
Jm = tensor(jmat(N_qubits/2, '-'), qeye(N_cavity))
a  = tensor(qeye(N_qubits + 1), destroy(N_cavity))
H_B_total = Jz + N_qubits/2

psi_init = tensor(basis(N_qubits+1, 0), basis(N_cavity, 0)) # بطارية مشحونة، تجويف بارد
rho_init = ket2dm(psi_init)

H_evals = np.sort(jmat(N_qubits/2, 'z').eigenenergies() + N_qubits/2)
def calc_ergo(state):
    rho_b = state.ptrace(0)
    evals = np.sort(rho_b.eigenenergies())[::-1]
    E_curr = expect(H_B_total.ptrace(0), rho_b)
    E_pass = np.sum(evals * H_evals)
    return max(0, E_curr - E_pass)

# ==========================================
# 3. دوال البيئة الحرارية المتغيرة زمنياً
# ==========================================
# دالة حساب متوسط عدد الفوتونات الحرارية كدالة في الزمن
def n_thermal(t, args):
    # درجة حرارة تتذبذب جيبياً
    T_t = T_avg + T_amp * np.sin(2 * np.pi * t / tau_cycle)
    T_t = max(T_t, 0.05) # منع الوصول للصفر المطلق لتفادي أخطاء القسمة
    return 1.0 / (np.exp(w_b / T_t) - 1.0)

# مؤثرات الانهيار (واحدة للامتصاص وأخرى للانبعاث)
def c_op_emission(t, args):
    return np.sqrt(gamma_th * (1 + n_thermal(t, args)))

def c_op_absorption(t, args):
    return np.sqrt(gamma_th * n_thermal(t, args))

c_ops = [
    [a, c_op_emission],        # فقدان الطاقة للبيئة
    [a.dag(), c_op_absorption] # اكتساب فوتونات حرارية عشوائية (ضجيج)
]

# ==========================================
# 4. تشغيل المحاكاة (المقارنة)
# ==========================================
# حالة 1: بدون فلتر (رنين)
print("Simulating No Filter (Resonance)...")
H_0 = w_b * (Jz + N_qubits/2) + w_b * a.dag() * a + g * (Jp * a + Jm * a.dag())
res_0 = mesolve(H_0, rho_init, tlist, c_ops, [])
ergo_0 = [calc_ergo(s) for s in res_0.states]

# حالة 2: الفلتر المثالي
print("Simulating Optimal Filter...")
delta_opt = g * np.sqrt(N_qubits / (2 * gamma_th)) # نستخدم معادلتنا السحرية!
w_p = w_b + delta_opt
H_opt = w_b * (Jz + N_qubits/2) + w_p * a.dag() * a + g * (Jp * a + Jm * a.dag())
res_opt = mesolve(H_opt, rho_init, tlist, c_ops, [])
ergo_opt = [calc_ergo(s) for s in res_opt.states]

# ==========================================
# 5. الرسم البياني لديناميكيا الحرارة
# ==========================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 2]})

# رسم تذبذب درجة الحرارة كمرجع
T_plot = [T_avg + T_amp * np.sin(2 * np.pi * t / tau_cycle) for t in tlist]
ax1.plot(tlist, T_plot, 'orange', linewidth=2)
ax1.set_ylabel('Lab Temp $T(t)$', fontsize=12)
ax1.set_title('Environment B: Periodic Thermal Cycle & Ergotropy Response', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.set_xticklabels([]) # إخفاء أرقام المحور السيني للرسم العلوي

# رسم الإرجوتروبي
ax2.plot(tlist, ergo_0, 'k--', linewidth=2, label=r'No Filter ($\Delta=0$)')
ax2.plot(tlist, ergo_opt, 'r-', linewidth=2.5, label=fr'Optimal Filter ($\Delta={delta_opt:.2f}$)')
ax2.set_xlabel('Time', fontsize=12)
ax2.set_ylabel('Residual Ergotropy', fontsize=12)
ax2.legend(loc='upper right', fontsize=12)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('thermal_cycle_protection.png', dpi=300)
plt.show()

print("✅ Simulation complete! Check the generated plot.")
