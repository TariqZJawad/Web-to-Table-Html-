import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# ==========================================
# 1. تفعيل "فلتر الذاكرة النفعي" (Parameters & Filter)
# ==========================================
N = 5             
w_b = 1.0         

# ------ التعديل الجوهري هنا ------
delta_filter = 0.3         # قوة الفلتر (عدم الرنين)
w_p = w_b + delta_filter   # تردد الدرع بعد تفعيل الفلتر
# ---------------------------------

g = 0.5           
gamma = 0.2       

sm = tensor(sigmam(), qeye(N))       
sz = tensor(sigmaz(), qeye(N))       
a = tensor(qeye(2), destroy(N))      

H = w_b * sm.dag() * sm + w_p * a.dag() * a + g * (sm.dag() * a + sm * a.dag())
H_B = w_b * sigmam().dag() * sigmam()

psi0 = tensor(basis(2, 0), basis(N, 0)) 
c_ops = [np.sqrt(gamma) * a]
tlist = np.linspace(0, 20, 500)

# ==========================================
# 2. دالة حساب الإرجوتروبي 
# ==========================================
def calc_ergotropy(state):
    rho_b = state.ptrace(0) 
    r_evals = np.sort(rho_b.eigenenergies())[::-1] 
    H_evals = np.sort(H_B.eigenenergies())         
    E_current = expect(H_B, rho_b)
    E_passive = np.sum(r_evals * H_evals)
    return E_current - E_passive

# ==========================================
# 3. المحاكاة (Stochastic vs Markovian)
# ==========================================
print("Applying Utilitarian Memory Filter...")

plt.figure(figsize=(10, 6))

for i in range(3):
    mc_result = mcsolve(H, psi0, tlist, c_ops, [], ntraj=1)
    states_list = mc_result.states[0] if isinstance(mc_result.states[0], list) else mc_result.states
    ergo_trajectory = [calc_ergotropy(state) for state in states_list]
    plt.plot(tlist, ergo_trajectory, alpha=0.4, label=f'Filtered Ergotropy (Jump {i+1})')

me_result = mesolve(H, psi0, tlist, c_ops, [])
ergo_avg = [calc_ergotropy(state) for state in me_result.states]

plt.plot(tlist, ergo_avg, 'k--', linewidth=2.5, label='Filtered Average (Markovian)')

# ==========================================
# 4. رسم النتائج
# ==========================================
plt.title('Protection of Ergotropy via Utilitarian Memory Filter', fontsize=14)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Ergotropy (Useful Energy)', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()
