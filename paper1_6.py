import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from scipy.integrate import simpson

print("🚀 Running Systematic Sweep: BLP and Ergotropy vs Detuning...")

# ==========================================
# 1. إعدادات المحاكاة (N=2 لتوضيح الظاهرة بدقة وسرعة)
# ==========================================
N_qubits = 2
N_cavity = 5
w_b = 1.0
g = 0.5
lam_decay = 0.2
gamma0_fixed = 0.8
tlist = np.linspace(0, 15.0, 150) # دقة زمنية أعلى لحساب المشتقة (BLP) بدقة

# المؤثرات
Jz = tensor(jmat(N_qubits/2, 'z'), qeye(N_cavity))
Jp = tensor(jmat(N_qubits/2, '+'), qeye(N_cavity))
Jm = tensor(jmat(N_qubits/2, '-'), qeye(N_cavity))
a  = tensor(qeye(N_qubits + 1), destroy(N_cavity))
H_B_total = Jz + N_qubits/2

# الحالات الابتدائية (لحساب مسافة الأثر)
psi1 = tensor(basis(N_qubits+1, 0), basis(N_cavity, 0)) # مشحونة
rho1 = ket2dm(psi1)
psi2 = tensor(basis(N_qubits+1, N_qubits), basis(N_cavity, 0)) # فارغة
rho2 = ket2dm(psi2)

# ==========================================
# 2. حلقة مسح الإزاحة (Delta Sweep)
# ==========================================
delta_vals = np.linspace(0.0, 2.0, 25)
blp_results = []
ergo_results = []

H_evals = np.sort(jmat(N_qubits/2, 'z').eigenenergies() + N_qubits/2)
def calc_ergo(state):
    rho_b = state.ptrace(0)
    evals = np.sort(rho_b.eigenenergies())[::-1]
    return max(0, expect(H_B_total.ptrace(0), rho_b) - np.sum(evals * H_evals))

for delta in delta_vals:
    w_p = w_b + delta
    H = w_b * (Jz + N_qubits/2) + w_p * a.dag() * a + g * (Jp * a + Jm * a.dag())
    
    def decay(t, args): return np.sqrt(args['gamma0'] * np.exp(-lam_decay * t))
    c_ops = [[a, decay]]
    
    # محاكاة الحالتين
    res1 = mesolve(H, rho1, tlist, c_ops, [], args={'gamma0': gamma0_fixed})
    res2 = mesolve(H, rho2, tlist, c_ops, [], args={'gamma0': gamma0_fixed})
    
    # 1. حساب المتوسط الزمني للإرجوتروبي
    ergo_traj = [calc_ergo(s) for s in res1.states]
    ergo_results.append(np.mean(ergo_traj))
    
    # 2. حساب مقياس BLP
    dist = [tracedist(res1.states[i].ptrace(0), res2.states[i].ptrace(0)) for i in range(len(tlist))]
    deriv = np.gradient(dist, tlist[1] - tlist[0])
    blp = simpson(np.maximum(deriv, 0), x=tlist)
    blp_results.append(blp)

# ==========================================
# 3. الرسم البياني المزدوج (مع المناطق الفيزيائية)
# ==========================================
fig, ax1 = plt.subplots(figsize=(10, 6))

color1 = 'tab:red'
ax1.set_xlabel(r'Detuning Filter ($\Delta$)', fontsize=14)
ax1.set_ylabel(r'Time-Averaged Ergotropy $\langle \epsilon \rangle$', color=color1, fontsize=14)
line1 = ax1.plot(delta_vals, ergo_results, 'ro-', linewidth=2.5, label='Ergotropy')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, alpha=0.3)

# المحور الثاني
ax2 = ax1.twinx()  
color2 = 'tab:blue'
ax2.set_ylabel('Non-Markovianity (BLP Measure)', color=color2, fontsize=14)
line2 = ax2.plot(delta_vals, blp_results, 'bs--', linewidth=2.5, label='BLP Measure')
ax2.tick_params(axis='y', labelcolor=color2)

# ==========================================
# الإضافة العبقرية: تظليل المناطق الفيزيائية
# ==========================================
theoretical_delta = g * np.sqrt(N_qubits / (2 * gamma0_fixed))
ax1.axvline(theoretical_delta, color='k', linestyle=':', linewidth=2, label=r'Theoretical $\Delta^*$')

# 1. Memory-Dominated Regime (المنطقة قبل التقاطع)
ax1.axvspan(0, theoretical_delta, color='gray', alpha=0.15)
ax1.text(theoretical_delta / 2, max(ergo_results)*0.1, 'Memory-Dominated\nRegime\n(Strong Backflow, High Loss)', 
         horizontalalignment='center', verticalalignment='center', fontsize=11, color='dimgray', weight='bold')

# 2. Detuning-Dominated Regime (المنطقة بعد التقاطع)
ax1.axvspan(theoretical_delta, max(delta_vals), color='gold', alpha=0.1)
ax1.text((theoretical_delta + max(delta_vals)) / 2, max(ergo_results)*0.1, 'Detuning-Dominated\nRegime\n(Virtual Photon Dressing)', 
         horizontalalignment='center', verticalalignment='center', fontsize=11, color='darkgoldenrod', weight='bold')

# جمع المفاتيح
lines = line1 + line2 + [plt.Line2D([0], [0], color='k', linestyle=':', linewidth=2)]
labels = ['Ergotropy', 'BLP Measure', r'Optimal Balance $\Delta^*$']
ax1.legend(lines, labels, loc='center right', fontsize=12)

plt.title(r'The Non-Markovianity Paradox ($N=2$ Qubits)', fontsize=16)
fig.tight_layout()
plt.savefig('blp_paradox_shaded.png', dpi=300)
plt.show()

