import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# --- 1. بناء النظام (نفس الهاميلتونيان السابق) ---
omega = 1.0   
zeta = 0.1    

sz = sigmaz(); sp = sigmap(); sm = sigmam(); I = qeye(2)

sz1 = tensor(sz, I); sp1 = tensor(sp, I); sm1 = tensor(sm, I)
sz2 = tensor(I, sz); sp2 = tensor(I, sp); sm2 = tensor(I, sm)

H0 = 0.5 * omega * sz1 + 0.5 * omega * sz2
H_int = zeta * (sp1 * sm2 + sm1 * sp2)
H_total = H0 + H_int

H_battery = 0.5 * omega * sz2

# --- 2. الحالة الابتدائية (الشاحن ممتلئ، البطارية فارغة) ---
psi_0 = tensor(basis(2, 0), basis(2, 1))
times = np.linspace(0, 50, 400)

# =====================================================================
# --- 3. البيئة المثبطة (معادلة ليندبلاد) ---
# جرب تغيير قيمة gamma لاختبار فرضيتك!
# gamma = 0.05  (تثبيط بطيء وطبيعي)
# gamma = 2.0   (البيك الهائل الذي اقترحته - تأثير زينو للتجميد!)
# =====================================================================
gamma = 2.00  

# مؤثرات الانهيار (Collapse Operators) - الشاحن والبطارية كلاهما يسربان طاقة للفراغ
c_ops = [np.sqrt(gamma) * sm1, np.sqrt(gamma) * sm2]

# --- 4. التطور الزمني المفتوح ---
# وضعنا c_ops داخل mesolve، الآن البيئة تدمر النظام!
result = mesolve(H_total, psi_0, times, c_ops, [H_battery])

battery_energy = result.expect[0]

# --- 5. الرسم البياني ---
plt.figure(figsize=(10, 5))
plt.plot(times, battery_energy, label=rf'Battery Energy ($\gamma = {gamma}$)', color='red', linewidth=2)

plt.axhline(y=0.5, color='gray', linestyle='--', label='Ideal Max Charge (0.5)')
plt.axhline(y=-0.5, color='gray', linestyle='--', label='Empty (-0.5)')

plt.title("Quantum Battery Charging in a Markovian Damping Environment")
plt.xlabel("Time (t)")
plt.ylabel("Energy")
plt.legend(loc='upper right')
plt.grid(True)
plt.show()
