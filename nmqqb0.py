import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# ==========================================
# 1. تعريف المعاملات الفيزيائية للنظام
# ==========================================
N = 5             # مستويات الطاقة للدرع الكمي (Pseudomode)
w_b = 1.0         # تردد البطارية الكمية
w_p = 1.0         # تردد الدرع (في حالة رنين مع البطارية لنقل الطاقة)
g = 0.5           # قوة التفاعل (الاقتران) بين البطارية والدرع
gamma = 0.2       # معدل تبدد الدرع (قوة العواصف الكونية التي تضربه)

# ==========================================
# 2. بناء المؤثرات الكمية (Operators)
# ==========================================
# مؤثرات البطارية (نظام ثنائي)
sm = tensor(sigmam(), qeye(N)) # مؤثر الهبوط للبطارية
sz = tensor(sigmaz(), qeye(N)) # مؤثر قياس طاقة البطارية

# مؤثرات الدرع (نظام بوزوني)
a = tensor(qeye(2), destroy(N)) # مؤثر الهبوط للدرع

# الهاميلتوني الكلي (البطارية + الدرع + تفاعلهما)
H = w_b * sm.dag() * sm + w_p * a.dag() * a + g * (sm.dag() * a + sm * a.dag())

# ==========================================
# 3. الحالة الابتدائية والبيئة العشوائية
# ==========================================
# نفترض أن البطارية مشحونة بالكامل (1)، والدرع فارغ (0)
psi0 = tensor(basis(2, 0), basis(N, 0)) 

# مؤثرات الانهيار (العواصف التي تضرب الدرع فقط، وليس البطارية)
c_ops = [np.sqrt(gamma) * a]

# الزمن من 0 إلى 20
tlist = np.linspace(0, 20, 500)

# ==========================================
# 4. محاكاة مونت كارلو والرسم (Stochastic Jumps)
# ==========================================
print("Running Monte Carlo Simulation for Quantum Battery...")

plt.figure(figsize=(10, 6))

# تشغيل 5 مسارات عشوائية منفصلة ورسم كل واحد منها مباشرة
for i in range(5):
    # ntraj=1 تعني محاكاة مسار كوني واحد في كل دورة
    mc_result = mcsolve(H, psi0, tlist, c_ops, [sm.dag() * sm], ntraj=1)
    # mc_result.expect[0] تحتوي الآن على 500 نقطة بالضبط
    plt.plot(tlist, mc_result.expect[0], alpha=0.3, label=f'Trajectory {i+1}')

# ==========================================
# 5. محاكاة المتوسط الماركوفي للمقارنة
# ==========================================
# استخدام mesolve للحصول على المتوسط الناعم
me_result = mesolve(H, psi0, tlist, c_ops, [sm.dag() * sm])

# رسم متوسط كل المسارات (الخط العريض المتقطع)
plt.plot(tlist, me_result.expect[0], 'k--', linewidth=2.5, label='Average (Markovian)')

plt.title('Quantum Battery Protection under Stochastic Cosmic Jumps', fontsize=14)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Battery Population (Charge)', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()
