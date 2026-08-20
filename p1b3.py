import numpy as np
import matplotlib.pyplot as plt
from qutip import *

print("🚀 Starting Final Environment B Simulation: Random Telegraph Noise (RTN)...")

# ==========================================
# 1. إعدادات النظام
# ==========================================
N_qubits = 2
N_cavity = 5
w_b = 1.0
g = 0.5
gamma0_fixed = 0.8
lam_decay = 0.2

# إعدادات ضجيج RTN
sigma_rtn = 0.3       # سعة الضجيج (قوة القفزات في التردد)
gamma_rtn = 0.5       # معدل القفزات العشوائية
num_realizations = 20 # عدد المسارات العشوائية لأخذ المتوسط
tlist = np.linspace(0, 15.0, 150)

# ==========================================
# 2. المؤثرات
# ==========================================
iden_b = qeye(N_qubits + 1)
iden_c = qeye(N_cavity)

Jz = tensor(jmat(N_qubits/2, 'z'), iden_c)
Jp = tensor(jmat(N_qubits/2, '+'), iden_c)
Jm = tensor(jmat(N_qubits/2, '-'), iden_c)
a  = tensor(iden_b, destroy(N_cavity))

H_b_only = jmat(N_qubits/2, 'z') + (N_qubits/2) * qeye(N_qubits+1)
H_evals = np.sort(H_b_only.eigenenergies())

def calc_ergo(state):
    rho_b = state.ptrace(0)
    evals = np.sort(rho_b.eigenenergies())[::-1]
    E_curr = expect(H_b_only, rho_b)
    E_pass = np.sum(evals * H_evals)
    return max(0, E_curr - E_pass)

def decay(t, args): 
    return np.sqrt(args['gamma0'] * np.exp(-lam_decay * t))
c_ops = [[a, decay]]
args_env = {'gamma0': gamma0_fixed}

psi_init = tensor(basis(N_qubits+1, 0), basis(N_cavity, 0))
rho_init = ket2dm(psi_init)

# ==========================================
# 3. دالة توليد مسار الضجيج العشوائي (RTN)
# ==========================================
def generate_rtn_signal(tlist, sigma, gamma):
    dt = tlist[1] - tlist[0]
    signal = np.zeros_like(tlist)
    current_state = 1
    for i in range(1, len(tlist)):
        # احتمال حدوث قفزة في التردد
        if np.random.rand() < gamma * dt:
            current_state *= -1
        signal[i] = sigma * current_state
    return signal

# ==========================================
# 4. محاكاة المسارات العشوائية
# ==========================================
ergo_avg_no_filter = np.zeros_like(tlist)
ergo_avg_opt_filter = np.zeros_like(tlist)

delta_opt = g * np.sqrt(N_qubits / (2 * gamma0_fixed)) + 0.2
w_p = w_b + delta_opt

print(f"Averaging over {num_realizations} random noise realizations...")

for i in range(num_realizations):
    if i % 5 == 0: print(f"  Running realization {i+1}/{num_realizations}...")
    
    # توليد ضجيج فريد لهذا المسار
    rtn_noise = generate_rtn_signal(tlist, sigma_rtn, gamma_rtn)
    
    # الهاميلتونيان المعتمد على الزمن (إضافة الضجيج إلى Jz)
    H_0 = [w_b * Jz + (N_qubits/2)*w_b*tensor(iden_b, iden_c) + w_b * a.dag() * a + g * (Jp * a + Jm * a.dag()),
           [Jz, rtn_noise]]
           
    H_opt = [w_b * Jz + (N_qubits/2)*w_b*tensor(iden_b, iden_c) + w_p * a.dag() * a + g * (Jp * a + Jm * a.dag()),
             [Jz, rtn_noise]]
             
    res_0 = mesolve(H_0, rho_init, tlist, c_ops, [], args=args_env)
    res_opt = mesolve(H_opt, rho_init, tlist, c_ops, [], args=args_env)
    
    ergo_avg_no_filter += np.array([calc_ergo(s) for s in res_0.states])
    ergo_avg_opt_filter += np.array([calc_ergo(s) for s in res_opt.states])

# حساب المتوسط النهائي
ergo_avg_no_filter /= num_realizations
ergo_avg_opt_filter /= num_realizations

# ==========================================
# 5. الرسم البياني
# ==========================================
fig, ax = plt.subplots(figsize=(9, 6))

ax.plot(tlist, ergo_avg_no_filter, 'k--', linewidth=2, label=r'No Filter ($\Delta=0$)')
ax.plot(tlist, ergo_avg_opt_filter, 'r-', linewidth=2.5, label=fr'Optimal Filter ($\Delta={delta_opt:.2f}$)')

ax.set_xlabel('Time', fontsize=14)
ax.set_ylabel('Ensemble-Averaged Ergotropy', fontsize=14)
ax.set_title('Robustness Against Random Telegraph Noise (Dephasing)', fontsize=15)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rtn_noise_protection.png', dpi=300)
plt.show()

print("✅ RTN Simulation complete! The battery's resilience is mapped.")
