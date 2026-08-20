import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# --- 1. تعريف الترددات (الفيزياء) ---
omega = 1.0
zeta = 0.5         # قوة التفاعل بين الشاحن والبطارية
lambda_mem = 0.1   # قوة التفاعل بين البطارية و"عقدة الذاكرة" (الدرع)
gamma = 0.01       # قوة تسريب الذاكرة للفراغ الماركوفي القاسي

# --- 2. بناء فضاء 3 كيوبتات (شاحن، بطارية، ذاكرة) ---
# المصفوفة الآن أصبحت 8x8
I = qeye(2); sz = sigmaz(); sp = sigmap(); sm = sigmam()

# الشاحن (الكيوبت الأول)
sz1 = tensor(sz, I, I); sp1 = tensor(sp, I, I); sm1 = tensor(sm, I, I)
# البطارية (الكيوبت الثاني)
sz2 = tensor(I, sz, I); sp2 = tensor(I, sp, I); sm2 = tensor(I, sm, I)
# الذاكرة - الفلتر (الكيوبت الثالث)
sz3 = tensor(I, I, sz); sp3 = tensor(I, I, sp); sm3 = tensor(I, I, sm)

# --- 3. الهاميلتونيان (هندسة التفاعلات) ---
H0 = 0.5 * omega * (sz1 + sz2 + sz3)

# الشاحن يشحن البطارية
H_charging = zeta * (sp1 * sm2 + sm1 * sp2)

# البطارية تتفاعل مع الذاكرة (هنا يحدث التدفق العكسي!)
H_memory = lambda_mem * (sp2 * sm3 + sm2 * sp3)

H_total = H0 + H_charging + H_memory

# طاقة البطارية فقط للرسم
H_battery = 0.5 * omega * sz2

# --- 4. الحالة الابتدائية والديناميكا ---
# شاحن ممتلئ (0)، بطارية فارغة (1)، ذاكرة فارغة (1)
psi_0 = tensor(basis(2, 0), basis(2, 1), basis(2, 1))
times = np.linspace(0, 50, 400)

# =========================================================
# السحر هنا: التبدد (الضجيج الماركوفي) يضرب "الذاكرة" فقط!
# البطارية محمية خلف الفلتر، والذاكرة هي التي تتلقى الضربات.
# =========================================================
c_ops = [ np.sqrt(gamma)*sm3]

# تشغيل المحاكاة
result = mesolve(H_total, psi_0, times, c_ops, [H_battery])
battery_energy = result.expect[0]

# --- 5. الرسم البياني ---
plt.figure(figsize=(10, 5))
plt.plot(times, battery_energy, label='Battery Energy (Memory Shield)', color='green', linewidth=2)

plt.axhline(y=0.5, color='gray', linestyle='--', label='Ideal Max (0.5)')
plt.axhline(y=-0.5, color='gray', linestyle='--', label='Empty (-0.5)')

plt.title("Quantum Battery Protected by Non-Markovian Memory Filter")
plt.xlabel("Time (t)")
plt.ylabel("Energy")
plt.legend(loc='upper right')
plt.grid(True)
plt.show()
