import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# --- 1. الباراميترات المثالية (تجويف بصري عالي العزل) ---
omega = 1.0
zeta = 0.5         # جسر شحن قوي وسريع
lambda_mem = 0.1   # فلتر الذاكرة النفعي
gamma = 0.01       # ضجيج ماركوفي يتسرب من الفلتر فقط

# بناء المؤثرات
sz = sigmaz(); sp = sigmap(); sm = sigmam(); I = qeye(2)

sz1 = tensor(sz, I, I); sp1 = tensor(sp, I, I); sm1 = tensor(sm, I, I)
sz2 = tensor(I, sz, I); sp2 = tensor(I, sp, I); sm2 = tensor(I, sm, I)
sz3 = tensor(I, I, sz); sp3 = tensor(I, I, sp); sm3 = tensor(I, I, sm)

# الهاميلتونيان
H0 = 0.5 * omega * (sz1 + sz2 + sz3)
H_charging = zeta * (sp1 * sm2 + sm1 * sp2)
H_memory = lambda_mem * (sp2 * sm3 + sm2 * sp3)
H_total = H0 + H_charging + H_memory

# --- 2. دالة حساب الإرجوتروبي للبطارية (من نظام 3 كيوبت) ---
def battery_ergotropy_3q(global_state):
    # التقفي الجزئي: نحتفظ بالكيوبت رقم 1 (البطارية) ونتجاهل 0 و 2
    rho_B = global_state.ptrace(1)
    
    # هاميلتونيان محلي للبطارية (مصفوفة 2x2)
    H_B_local = 0.5 * omega * sigmaz()
    
    # حساب القيم الذاتية
    rho_evals = np.sort(rho_B.eigenenergies())[::-1] # تنازلي
    H_evals = np.sort(H_B_local.eigenenergies())     # تصاعدي
    
    current_energy = expect(H_B_local, rho_B)
    passive_energy = np.sum(rho_evals * H_evals)
    
    return current_energy - passive_energy

# --- 3. الحالة الابتدائية والديناميكا ---
# الشاحن (ممتلئ 0)، البطارية (فارغة 1)، الذاكرة (فارغة 1)
psi_0 = tensor(basis(2, 0), basis(2, 1), basis(2, 1))
times = np.linspace(0, 50, 400)

# البيئة تهاجم الكيوبت الثالث فقط (الدرع)
c_ops = [np.sqrt(gamma) * sm3]

# تركنا قائمة المؤثرات فارغة [] لكي نجمع الحالات (states) عند كل لحظة زمنية
result = mesolve(H_total, psi_0, times, c_ops, [])

# --- 4. استخراج وحساب البيانات للمنحنيات ---
battery_energy_list = []
ergotropy_list = []

H_B_global = 0.5 * omega * sz2

for state in result.states:
    # حساب الطاقة الكلية للبطارية
    energy = expect(H_B_global, state)
    battery_energy_list.append(energy)
    
    # حساب الشغل الصافي (الإرجوتروبي)
    ergo = battery_ergotropy_3q(state)
    ergotropy_list.append(ergo)

# --- 5. الرسم البياني الاحترافي للورقة البحثية ---
plt.figure(figsize=(10, 5))

# رسم الطاقة الكلية للبطارية (المنحنى الأخضر الذي أنقذناه)
plt.plot(times, battery_energy_list, label=r'Total Energy $E_B(t)$', color='green', linewidth=2)

# رسم الإرجوتروبي (المنحنى الأحمر)
plt.plot(times, ergotropy_list, label=r'Ergotropy $\mathcal{E}(t)$', color='red', linestyle='-', linewidth=2.5)

plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
plt.axhline(y=-0.5, color='gray', linestyle='--', alpha=0.5)

plt.title("Energy vs Ergotropy in a Protected 3-Qubit Quantum Battery")
plt.xlabel("Time (t)")
plt.ylabel("Value")
plt.legend(loc='upper right')
plt.grid(True)
plt.show()
