import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from scipy.integrate import simpson

# ==========================================
# 1. إعدادات المصفوفة الجماعية والفلتر
# ==========================================
N_qubits = 2          # نبدأ بـ 2 كيوبت لتسريع الخريطة الحرارية (يمكنك رفعها لـ 4 لاحقاً)
N_cavity = 4          # مستويات التجويف
w_b = 1.0             # تردد البطارية
g = 0.5               # قوة التفاعل الجماعي
lam_decay = 0.2       # معدل الاضمحلال النووي

# تعريف المؤثرات الجماعية (Dicke Basis)
Jz = tensor(jmat(N_qubits/2, 'z'), qeye(N_cavity))
Jp = tensor(jmat(N_qubits/2, '+'), qeye(N_cavity))
Jm = tensor(jmat(N_qubits/2, '-'), qeye(N_cavity))
a  = tensor(qeye(N_qubits+1), destroy(N_cavity))
H_B_total = Jz + N_qubits/2

# ==========================================
# 2. الحالات الابتدائية لمقياس BLP
# ==========================================
# نحتاج لحالتين مختلفتين لنرى كيف تتغير "المسافة" بينهما
# الحالة 1: البطاريات مشحونة بالكامل
psi1_b = basis(N_qubits+1, 0) # في فضاء Dicke
psi1 = tensor(psi1_b, basis(N_cavity, 0))
rho1_init = ket2dm(psi1)

# الحالة 2: البطاريات فارغة تماماً
psi2_b = basis(N_qubits+1, N_qubits)
psi2 = tensor(psi2_b, basis(N_cavity, 0))
rho2_init = ket2dm(psi2)

tlist = np.linspace(0, 20, 200)

# ==========================================
# 3. الدوال الأساسية (الإرجوتروبي و الاضمحلال)
# ==========================================
def nuclear_decay(t, args):
    return np.sqrt(args['gamma0'] * np.exp(-lam_decay * t))

def calc_collective_ergotropy(state):
    rho_b = state.ptrace(0)
    evals = np.sort(rho_b.eigenenergies())[::-1]
    H_evals = np.sort(jmat(N_qubits/2, 'z').eigenenergies() + N_qubits/2)
    E_current = expect(H_B_total.ptrace(0), rho_b)
    E_passive = np.sum(evals * H_evals)
    return max(0, E_current - E_passive)

# ==========================================
# 4. محرك المحاكاة للورقة الأولى (مسافة الأثر)
# ==========================================
def simulate_blp_and_ergotropy(delta, gamma0):
    w_p = w_b + delta
    H = w_b * (Jz + N_qubits/2) + w_p * a.dag() * a + g * (Jp * a + Jm * a.dag())
    c_ops = [[a, nuclear_decay]]
    args = {'gamma0': gamma0}
    
    # حل المعادلة للحالتين
    res1 = mesolve(H, rho1_init, tlist, c_ops, [], args=args)
    res2 = mesolve(H, rho2_init, tlist, c_ops, [], args=args)
    
    # حساب الإرجوتروبي للحالة المشحونة (rho1)
    ergo = [calc_collective_ergotropy(s) for s in res1.states]
    
    # حساب مسافة الأثر بين الحالتين D(rho1, rho2)
    trace_dist = [tracedist(res1.states[i].ptrace(0), res2.states[i].ptrace(0)) for i in range(len(tlist))]
    
    # حساب مقياس BLP (تكامل المشتقة الموجبة فقط)
    dt = tlist[1] - tlist[0]
    deriv = np.gradient(trace_dist, dt)
    blp_measure = simpson(np.maximum(deriv, 0), tlist) # نأخذ الأجزاء الموجبة فقط
    
    return ergo, trace_dist, blp_measure

# ==========================================
# 5. تشغيل واختبار قيمتين للإزاحة
# ==========================================
print("Running simulations for Markovian vs Non-Markovian regimes...")
gamma_test = 1.0

# 1. بدون فلتر (رنين - ماركوفي تقريباً)
ergo_0, dist_0, blp_0 = simulate_blp_and_ergotropy(delta=0.0, gamma0=gamma_test)

# 2. مع فلتر الإزاحة النفعي (غير ماركوفي)
ergo_opt, dist_opt, blp_opt = simulate_blp_and_ergotropy(delta=0.8, gamma0=gamma_test)

# ==========================================
# 6. الرسم البياني للورقة (Figure 1 & 2)
# ==========================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# رسم مسافة الأثر (مؤشر الذاكرة)
ax1.plot(tlist, dist_0, 'k--', label=fr'No Filter ($\Delta=0$), BLP={blp_0:.3f}')
ax1.plot(tlist, dist_opt, 'b-', linewidth=2, label=f'Optimal Filter ($\Delta=0.8$), BLP={blp_opt:.3f}')
ax1.set_title('Trace Distance Dynamics (Information Backflow)', fontsize=14)
ax1.set_ylabel('$D(\\rho_1, \\rho_2)$', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)

# رسم الإرجوتروبي المتبقي
ax2.plot(tlist, ergo_0, 'k--', label='Ergotropy (No Filter)')
ax2.plot(tlist, ergo_opt, 'r-', linewidth=2, label='Ergotropy (Optimal Filter)')
ax2.set_title('Collective Ergotropy Protection', fontsize=14)
ax2.set_xlabel('Time', fontsize=12)
ax2.set_ylabel('Ergotropy', fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ==========================================
# ملاحظة: كود الخريطة الحرارية (Heatmap) جاهز
# سأعطيك إياه في الخطوة القادمة لتجنب تجميد حاسوبك!
# ==========================================
