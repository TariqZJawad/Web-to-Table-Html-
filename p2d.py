import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from scipy.integrate import simpson

# 1. إعداد المعاملات الأساسية 
Nc = 4; Nm = 3; Np = 3; Nq = 2  
w_c = 1.0; w_m = 1.0; w_p = 0.1; w_q = 1.0        
lam = 0.2; g_a = 0.15       
kappa_c = 0.05; kappa_m = 0.05; kappa_p = 0.01; gamma_q = 0.02
tlist = np.linspace(0, 40, 150) 
gb_scan = np.linspace(0.0, 0.2, 10)

# 2. بناء المؤثرات
a  = tensor(destroy(Nc), qeye(Nm), qeye(Np), qeye(2), qeye(2))
b  = tensor(qeye(Nc), destroy(Nm), qeye(Np), qeye(2), qeye(2))
p  = tensor(qeye(Nc), qeye(Nm), destroy(Np), qeye(2), qeye(2))
sz1 = tensor(qeye(Nc), qeye(Nm), qeye(Np), sigmaz(), qeye(2))
sz2 = tensor(qeye(Nc), qeye(Nm), qeye(Np), qeye(2), sigmaz())
sx1 = tensor(qeye(Nc), qeye(Nm), qeye(Np), sigmax(), qeye(2))
sx2 = tensor(qeye(Nc), qeye(Nm), qeye(Np), qeye(2), sigmax())
Jz = 0.5 * (sz1 + sz2)
Jx = 0.5 * (sx1 + sx2)

c_ops = [
    np.sqrt(kappa_c) * a, 
    np.sqrt(kappa_m) * b, 
    np.sqrt(kappa_p) * p,
    tensor(qeye(Nc), qeye(Nm), qeye(Np), np.sqrt(gamma_q)*destroy(2), qeye(2)),
    tensor(qeye(Nc), qeye(Nm), qeye(Np), qeye(2), np.sqrt(gamma_q)*destroy(2))
]

# 3. دالة حساب الـ BLP بالطريقة الفيزيائية الصحيحة (Orthogonal Qubit States)
def calculate_correct_blp(delta, g_b):
    H_CMM = (w_c + delta)*a.dag()*a + w_m*b.dag()*b + w_p*p.dag()*p \
            + g_a*(a+a.dag())*(b+b.dag()) + g_b*(b+b.dag())*(p+p.dag())
    H_full = H_CMM + w_q*Jz + 2*lam*(a+a.dag())*Jx
    
    # حالتان متعامدتان للبطارية (|1,0> و |0,1>) مع ثبات بيئة التجويف الممتلئ
    psi1 = tensor(basis(Nc, 2), basis(Nm,0), basis(Np,0), basis(2,1), basis(2,0))
    psi2 = tensor(basis(Nc, 2), basis(Nm,0), basis(Np,0), basis(2,0), basis(2,1))
    
    res1 = mesolve(H_full, ket2dm(psi1), tlist, c_ops, [])
    res2 = mesolve(H_full, ket2dm(psi2), tlist, c_ops, [])
    
    trace_dists = [tracedist(res1.states[i].ptrace([3,4]), res2.states[i].ptrace([3,4])) for i in range(len(tlist))]
    derivs = np.gradient(trace_dists, tlist[1]-tlist[0])
    return simpson(np.maximum(derivs, 0), x=tlist)

# 4. التنفيذ المباشر لرسم الشكل الثاني فقط
print("🔍 Computing Correct BLP Measure (Takes 2-3 minutes)...")

# ضع هنا الدلتا المثالية التي ظهرت لك في نتائج ملف الـ CSV
opt_delta = -0.056 # (قم بتغيير هذا الرقم إذا كانت الدلتا المثالية في ملفك مختلفة)

blp_slice = []
for gb in gb_scan:
    blp = calculate_correct_blp(opt_delta, gb)
    blp_slice.append(blp)

# رسم الشكل الثاني المصحح
fig2, ax = plt.subplots(figsize=(8, 6))
color = 'tab:blue'
ax.plot(gb_scan, blp_slice, 'bs--', lw=2, label=r'BLP measure $\mathcal{N}$')
ax.set_xlabel(r'Phonon Coupling Strength $g_b$')
ax.set_ylabel(r'Non-Markovianity $\mathcal{N}_{BLP}$', color=color)
ax.tick_params(axis='y', labelcolor=color)
plt.title(r'Phonon-Induced Memory (Corrected BLP)')
fig2.tight_layout()
plt.savefig('Fig2_NonMarkovian.png')

print("✅ Done! Figure 2 is corrected and ready for the paper.")
