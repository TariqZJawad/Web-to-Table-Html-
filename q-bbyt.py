import numpy as np
import matplotlib.pyplot as plt
from qutip import basis, sigmaz, sigmax, mesolve

# --- 1. بناء الهاميلتونيان (الكلي) ---
omega = 1.0  # تردد طاقة البطارية
H0 = 0.5 * omega * sigmaz()  # الهاميلتونيان الحر (الطاقة الداخلية للبطارية)

g = 0.5  # قوة المجال الخارجي (سرعة الشحن)
H_int = g * sigmax()  # هاميلتونيان التفاعل (المصدر الذي يشحن البطارية)

# الهاميلتونيان الكلي = طاقة البطارية + طاقة الشحن
H_total = H0 + H_int

# --- 2. الحالة الابتدائية ---
# سنبدأ ببطارية فارغة تماماً (الحالة الأرضية)
psi_0 = basis(2, 1)

# --- 3. إعداد الزمن والتطور الديناميكي ---
# سنراقب البطارية لمدة 15 ثانية، ونأخذ 200 لقطة (نقطة زمنية)
times = np.linspace(0, 15, 200)

# استخدام mesolve لحساب تطور النظام. 
# المتغير الأخير [H0] يخبر الدالة: "احسبي لي قيمة H0 (الطاقة) عند كل نقطة زمنية"
result = mesolve(H_total, psi_0, times, [], [H0])

# استخراج قائمة قيم الطاقة من النتيجة
energy_over_time = result.expect[0]

# --- 4. رسم النتائج البيانية ---
plt.figure(figsize=(10, 5))
plt.plot(times, energy_over_time, label='Internal Energy $E(t)$', color='blue', linewidth=2)

# رسم خطوط مرجعية للبطارية الممتلئة والفارغة
plt.axhline(y=0.5, color='red', linestyle='--', label='Fully Charged (0.5)')
plt.axhline(y=-0.5, color='green', linestyle='--', label='Empty (-0.5)')

plt.title("Quantum Battery Charging Dynamics (Closed System)")
plt.xlabel("Time (t)")
plt.ylabel("Energy")
plt.legend(loc='upper right')
plt.grid(True)
plt.show()
