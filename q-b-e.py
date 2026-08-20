import numpy as np
import matplotlib.pyplot as plt
from qutip import basis, sigmaz, sigmax, mesolve

# --- 1. بناء الهاميلتونيان (الكلي) ---
omega = 1.0  
H0 = 0.5 * omega * sigmaz()  
g = 0.5  
H_int = g * sigmax()  
H_total = H0 + H_int

# --- 2. دالة حساب الإرجوتروبي (The Gap Code) ---
def calculate_ergotropy(rho, H0):
    # أ. حساب الطاقة الحالية Tr(rho * H0)
    current_energy = (rho * H0).tr()
    
    # ب. استخراج القيم الذاتية وترتيبها
    # مصفوفة الكثافة: دالة eigenenergies() تستخرج القيم، ونرتبها تنازلياً باستخدام [::-1]
    r = np.sort(rho.eigenenergies())[::-1]
    # الهاميلتونيان: نستخرج مستويات الطاقة ونرتبها تصاعدياً
    e = np.sort(H0.eigenenergies())
    
    # ج. حساب طاقة الحالة الخاملة (ضرب المصفوفات وجمعها)
    passive_energy = np.sum(r * e)
    
    # د. الإرجوتروبي هو الفرق بينهما
    return current_energy - passive_energy

# --- 3. الحالة الابتدائية والتطور ---
psi_0 = basis(2, 1)
# يجب تحويل المتجه إلى "مصفوفة كثافة" لكي تعمل دالة الإرجوتروبي بشكل صحيح
rho_0 = psi_0 * psi_0.dag()

times = np.linspace(0, 15, 200)

# التعديل هنا: تركنا القوائم فارغة [] لكي نطلب من mesolve أن تعطينا 
# حالة النظام (rho) بالكامل عند كل لحظة زمنية، بدلاً من مجرد رقم الطاقة
result = mesolve(H_total, rho_0, times, [], [])

# --- 4. استخراج البيانات لحساب المنحنيات ---
energy_list = []
ergotropy_list = []

for rho_t in result.states:
    # نحسب الطاقة العادية
    e_t = (rho_t * H0).tr()
    # نحسب الإرجوتروبي باستخدام الدالة التي صنعناها
    ergo_t = calculate_ergotropy(rho_t, H0)
    
    # نضيف الأرقام للقوائم (نأخذ الجزء الحقيقي .real فقط لتجنب أي أخطاء رياضية)
    energy_list.append(e_t.real)
    ergotropy_list.append(ergo_t.real)

# --- 5. الرسم البياني ---
plt.figure(figsize=(10, 5))
plt.plot(times, energy_list, label='Internal Energy $E(t)$', color='blue')
plt.plot(times, ergotropy_list, label='Ergotropy $\mathcal{E}(t)$', color='purple', linestyle='--')

plt.axhline(y=-0.5, color='green', linestyle=':', label='Empty Energy (-0.5)')

plt.title("Energy vs Ergotropy (Closed Ideal System)")
plt.xlabel("Time (t)")
plt.ylabel("Energy / Ergotropy")
plt.legend(loc='upper right')
plt.grid(True)
plt.show()
	
