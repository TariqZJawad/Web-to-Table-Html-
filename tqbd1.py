import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# --- 1. بناء النظام ---
omega = 1.0   
zeta = 0.1    

sz = sigmaz(); sp = sigmap(); sm = sigmam(); I = qeye(2)

sz1 = tensor(sz, I); sp1 = tensor(sp, I); sm1 = tensor(sm, I)
sz2 = tensor(I, sz); sp2 = tensor(I, sp); sm2 = tensor(I, sm)

H0 = 0.5 * omega * sz1 + 0.5 * omega * sz2
H_int = zeta * (sp1 * sm2 + sm1 * sp2)
H_total = H0 + H_int

# --- 2. دالة حساب الإرجوتروبي للبطارية ---
def battery_ergotropy(global_state):
    # استخراج حالة البطارية فقط (الكيوبت الثاني دليله 1) باستخدام ptrace
    rho_B = global_state.ptrace(1)
    
    # الهاميلتونيان المحلي للبطارية (2x2)
    H_B = 0.5 * omega * sigmaz()
    
    # القيم الذاتية لمصفوفة كثافة البطارية (ترتيب تنازلي)
    rho_evals = np.sort(rho_B.eigenenergies())[::-1]
    
    # القيم الذاتية لهاميلتونيان البطارية (ترتيب تصاعدي)
    H_evals = np.sort(H_B.eigenenergies())
    
    # الطاقة الحالية للبطارية
    current_energy = expect(H_B, rho_B)
    
    # طاقة الحالة الخاملة (Passive State Energy)
    passive_energy = np.sum(rho_evals * H_evals)
    
    # الإرجوتروبي هو الفرق
    return current_energy - passive_energy

# --- 3. الحالة الابتدائية والديناميكا ---
psi_0 = tensor(basis(2, 0), basis(2, 1))
times = np.linspace(0, 50, 400)

# هذه المرة لم نضع H_battery في القوس الأخير، لأننا نريد استخراج "الحالة الكمية الكاملة" عند كل ثانية
result = mesolve(H_total, psi_0, times, [], [])

# --- 4. استخراج البيانات للمنحنيات ---
battery_energy_list = []
ergotropy_list = []

for state in result.states:
    # حساب الطاقة
    H_B_global = 0.5 * omega * sz2
    energy = expect(H_B_global, state)
    battery_energy_list.append(energy)
    
    # حساب الإرجوتروبي
    ergo = battery_ergotropy(state)
    ergotropy_list.append(ergo)

# --- 5. الرسم البياني الاحترافي ---
plt.figure(figsize=(10, 5))

# رسم الطاقة (أزرق)
plt.plot(times, battery_energy_list, label=r'Total Energy $E_B(t)$', color='blue', linewidth=2)

# رسم الإرجوتروبي (أحمر)
plt.plot(times, ergotropy_list, label=r'Ergotropy $\mathcal{E}(t)$', color='red', linestyle='-', linewidth=2.5)

plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
plt.axhline(y=-0.5, color='gray', linestyle='--', alpha=0.5)

plt.title("Energy vs Ergotropy in a Two-Qubit Quantum Battery")
plt.xlabel("Time (t)")
plt.ylabel("Value")
plt.legend(loc='upper right')
plt.grid(True)
plt.show()
