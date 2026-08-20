import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# ==========================================
# 1. المعاملات والمؤثرات (Parameters & Operators)
# ==========================================
N = 5             # مستويات طاقة الدرع البوزوني (Pseudomode)
w_b = 1.0         # تردد البطارية
w_p = 1.0         # تردد الدرع (في حالة رنين)
g = 0.5           # قوة الاقتران (تبادل الطاقة بين البطارية والدرع)
gamma = 0.2       # معدل التبدد (قوة العواصف الكونية التي تضرب الدرع)

# تعريف المؤثرات
sm = tensor(sigmam(), qeye(N))       # مؤثر الهبوط للبطارية
sz = tensor(sigmaz(), qeye(N))       # مؤثر شحنة البطارية
a = tensor(qeye(2), destroy(N))      # مؤثر الهدم للدرع

# الهاميلتوني الكلي للمحاكاة (بطارية + درع + تفاعل)
H = w_b * sm.dag() * sm + w_p * a.dag() * a + g * (sm.dag() * a + sm * a.dag())

# هاميلتوني البطارية فقط (لحساب الشغل المفيد منها حصراً)
H_B = w_b * sigmam().dag() * sigmam()

# الحالة الابتدائية: البطارية مشحونة (0) والدرع فارغ (0)
psi0 = tensor(basis(2, 0), basis(N, 0)) 

# مؤثرات الانهيار: العاصفة تضرب الدرع فقط (ليست البطارية مباشرة)
c_ops = [np.sqrt(gamma) * a]

# الزمن من 0 إلى 20
tlist = np.linspace(0, 20, 500)

# ==========================================
# 2. دالة حساب الإرجوتروبي (Ergotropy)
# ==========================================
def calc_ergotropy(state):
    # 1. عزل البطارية عن الدرع (اقتطاع التجويف للحصول على مصفوفة البطارية فقط)
    rho_b = state.ptrace(0) 
    
    # 2. حساب القيم الذاتية وترتيبها
    r_evals = np.sort(rho_b.eigenenergies())[::-1] # قيم مصفوفة الكثافة (تنازلي)
    H_evals = np.sort(H_B.eigenenergies())         # قيم طاقة البطارية (تصاعدي)
    
    # 3. حساب الطاقة الكلية للبطارية
    E_current = expect(H_B, rho_b)
    
    # 4. حساب طاقة الحالة الخاملة (Passive State) التي لا يمكن استخراج شغل منها
    E_passive = np.sum(r_evals * H_evals)
    
    # 5. الإرجوتروبي (الشغل الصافي والمفيد) هو الفرق بينهما
    return E_current - E_passive

# ==========================================
# 3. المحاكاة واستخراج الشغل (Simulation)
# ==========================================
print("Running Monte Carlo and calculating Ergotropy...")

plt.figure(figsize=(10, 6))

# تشغيل 3 مسارات عشوائية لتوضيح القفزات الكمية
for i in range(3):
    # محاكاة مسار واحد في كل مرة للحصول على الحالات (States)
    mc_result = mcsolve(H, psi0, tlist, c_ops, [], ntraj=1)
    
    # معالجة ذكية لاختلاف إصدارات QuTiP للحصول على قائمة الحالات بدقة
    states_list = mc_result.states[0] if isinstance(mc_result.states[0], list) else mc_result.states
    
    # حساب الإرجوتروبي لكل نقطة زمنية في هذا المسار
    ergo_trajectory = [calc_ergotropy(state) for state in states_list]
    plt.plot(tlist, ergo_trajectory, alpha=0.4, label=f'Ergotropy (Jump {i+1})')

# حساب متوسط الإرجوتروبي للحالة الماركوفية (mesolve) للمقارنة
print("Running Master Equation (Markovian Average)...")
me_result = mesolve(H, psi0, tlist, c_ops, [])
ergo_avg = [calc_ergotropy(state) for state in me_result.states]

plt.plot(tlist, ergo_avg, 'k--', linewidth=2.5, label='Average Ergotropy (Markovian)')

# ==========================================
# 4. رسم النتائج (Plotting)
# ==========================================
plt.title('Extractable Work (Ergotropy) from Quantum Battery', fontsize=14)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Ergotropy (Useful Energy)', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()
