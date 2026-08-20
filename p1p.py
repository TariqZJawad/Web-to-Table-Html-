import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from scipy.integrate import simpson

# إعدادات النشر العلمي
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'lines.linewidth': 2.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

print("="*60)
print("🔍 Generating Figure 13: The Non-Markovian Paradox Plot")
print("="*60)

# معاملات المحاكاة (لنظام N=2 لتسريع الحساب وإيضاح الفكرة)
N_qubits = 2
N_cavity = 4
w_b = 1.0
g = 0.5
gamma0 = 1.0
lam_decay = 0.2
tlist = np.linspace(0, 15, 100)

delta_vals = np.linspace(0.0, 1.5, 15) # مسح قيم الإزاحة
ergo_vals = []
blp_vals = []

# المؤثرات
Jz = tensor(jmat(N_qubits/2, 'z'), qeye(N_cavity))
Jp = tensor(jmat(N_qubits/2, '+'), qeye(N_cavity))
Jm = tensor(jmat(N_qubits/2, '-'), qeye(N_cavity))
a  = tensor(qeye(N_qubits + 1), destroy(N_cavity))
H_B_total = Jz + N_qubits/2

rho1_init = ket2dm(tensor(basis(N_qubits+1, 0), basis(N_cavity, 0)))
rho2_init = ket2dm(tensor(basis(N_qubits+1, N_qubits), basis(N_cavity, 0)))
H_evals = np.sort(jmat(N_qubits/2, 'z').eigenenergies() + N_qubits/2)

def calc_ergo(state):
    rho_b = state.ptrace(0)
    evals = np.sort(rho_b.eigenenergies())[::-1]
    E_pass = np.sum(evals * H_evals)
    return max(0.0, expect(H_B_total.ptrace(0), rho_b) - E_pass)

def decay(t, args): return np.sqrt(args['gamma0'] * np.exp(-lam_decay * t))
c_ops = [[a, decay]]

# حساب الديناميكا لكل قيمة لـ Delta
for delta in delta_vals:
    print(f"Calculating for Delta = {delta:.2f}...")
    H = w_b * (Jz + N_qubits/2) + (w_b + delta) * a.dag() * a + g * (Jp * a + Jm * a.dag())
    
    res1 = mesolve(H, rho1_init, tlist, c_ops, [], args={'gamma0': gamma0})
    res2 = mesolve(H, rho2_init, tlist, c_ops, [], args={'gamma0': gamma0})
    
    # حساب الإرجوتروبي المتبقي
    ergo_vals.append(calc_ergo(res1.states[-1]))
    
    # حساب مسافة المتتبع والـ BLP
    dist = [tracedist(res1.states[i].ptrace(0), res2.states[i].ptrace(0)) for i in range(len(tlist))]
    derivs = np.gradient(dist, tlist[1] - tlist[0])
    blp = simpson(np.maximum(derivs, 0), x=tlist)
    blp_vals.append(blp)

# رسم الشكل المزدوج (المفارقة)
fig, ax1 = plt.subplots(figsize=(7, 5))

color = 'tab:red'
ax1.set_xlabel(r'Detuning Parameter ($\Delta$)')
ax1.set_ylabel(r'Residual Ergotropy $\mathcal{E}_{res}$', color=color)
ax1.plot(delta_vals, ergo_vals, 'ro-', label=r'Ergotropy $\mathcal{E}_{res}$')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(alpha=0.3, linestyle='--')

ax2 = ax1.twinx()  # محور صادي ثاني للـ BLP
color = 'tab:blue'
ax2.set_ylabel(r'Non-Markovianity (BLP $\mathcal{N}$)', color=color)
ax2.plot(delta_vals, blp_vals, 'bs--', label=r'BLP $\mathcal{N}$')
ax2.tick_params(axis='y', labelcolor=color)

# إضافة خط يمثل الإزاحة المثلى
delta_opt_th = g * np.sqrt(N_qubits / (2 * gamma0)) + 0.2
ax1.axvline(delta_opt_th, color='k', linestyle=':', linewidth=2, label=r'Theoretical $\Delta^*$')

# ترتيب مفاتيح الرسم
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right')

plt.title('The Non-Markovian Paradox Plot')
plt.tight_layout()
plt.savefig('Figure_13_Paradox_Plot.png')
plt.close()

print("\n✅ Done! Figure_13_Paradox_Plot.png has been saved.")
