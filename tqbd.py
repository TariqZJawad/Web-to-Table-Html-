import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# --- 1. بناء النظام (نفس الكود السابق) ---
omega = 1.0   
zeta = 0.1    

sz = sigmaz(); sp = sigmap(); sm = sigmam(); I = qeye(2)

sz1 = tensor(sz, I); sp1 = tensor(sp, I); sm1 = tensor(sm, I)
sz2 = tensor(I, sz); sp2 = tensor(I, sp); sm2 = tensor(I, sm)

H0 = 0.5 * omega * sz1 + 0.5 * omega * sz2
H_int = zeta * (sp1 * sm2 + sm1 * sp2)
H_total = H0 + H_int

# طاقة البطارية فقط (نحتاجها للرسم)
H_battery = 0.5 * omega * sz2

# --- 2. الحالة الابتدائية (اللحظة صفر) ---
# الشاحن ممتلئ |e> (0) والبطارية فارغة |g> (1)
psi_0 = tensor(basis(2, 0), basis(2, 1))

# --- 3. التطور الزمني (الديناميكا) ---
# سنراقب النظام لمدة 50 ثانية
times = np.linspace(0, 50, 400)

# نستخدم mesolve لحل المعادلة، ونطلب منها استخراج (طاقة البطارية) عند كل نقطة زمنية
result = mesolve(H_total, psi_0, times, [], [H_battery])

battery_energy = result.expect[0]

# --- 4. الرسم البياني ---
plt.figure(figsize=(10, 5))
plt.plot(times, battery_energy, label=r'Battery Energy $E_B(t)$', color='blue', linewidth=2)

plt.axhline(y=0.5, color='red', linestyle='--', label='Fully Charged (0.5)')
plt.axhline(y=-0.5, color='green', linestyle='--', label='Empty (-0.5)')

plt.title("Two-Qubit Quantum Battery Charging (No External Environment)")
plt.xlabel("Time (t)")
plt.ylabel("Energy")
plt.legend(loc='upper right')
plt.grid(True)
plt.show()
