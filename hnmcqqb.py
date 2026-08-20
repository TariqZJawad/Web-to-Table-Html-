import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# ==========================================
# 1. معاملات النظام والبيئة الحرارية
# ==========================================
N = 5                  # مستويات طاقة الدرع البوزوني
w_b = 1.0              # تردد البطارية
delta_filter = 0.3     # قوة "فلتر الذاكرة النفعي" الذي صممناه بالأمس
w_p = w_b + delta_filter # تردد الدرع (مزاح قليلاً بفضل الفلتر)
g = 0.5                # قوة الاقتران بين البطارية والدرع
gamma = 0.2            # معدل تفاعل الدرع مع البيئة

# ------ الإضافة الجديدة: البيئة الحرارية ------
n_th = 0.5             # متوسط عدد الفوتونات الحرارية في البيئة (T > 0)
# ----------------------------------------------

# تعريف المؤثرات
sm = tensor(sigmam(), qeye(N))       # هبوط البطارية
sz = tensor(sigmaz(), qeye(N))       # شحنة البطارية
a = tensor(qeye(2), destroy(N))      # هدم الدرع (فقدان فوتون)
adag = tensor(qeye(2), create(N))    # بناء الدرع (اكتساب فوتون)

# الهاميلتوني الكلي للبطارية والدرع
H = w_b * sm.dag() * sm + w_p * a.dag() * a + g * (sm.dag() * a + sm * a.dag())

# هاميلتوني البطارية فقط (لحساب الشغل)
H_B = w_b * sigmam().dag() * sigmam()

# الحالة الابتدائية: بطارية مشحونة (0) ودرع فارغ (0)
psi0 = tensor(basis(2, 0), basis(N, 0)) 

# ==========================================
# 2. مؤثرات الانهيار الحراري (Thermal Jump Operators)
# ==========================================
# في البيئة الحرارية، لدينا نوعان من العواصف العشوائية تضرب الدرع:
c_ops = [
    np.sqrt(gamma * (1 + n_th)) * a,  # 1. فقدان الطاقة للبيئة (Emission)
    np.sqrt(gamma * n_th) * adag      # 2. امتصاص طاقة عشوائية من الحرارة (Absorption)
]

tlist = np.linspace(0, 20, 500)

# ==========================================
# 3. دالة حساب الإرجوتروبي (الشغل الصافي)
# ==========================================
def calc_ergotropy(state):
    rho_b = state.ptrace(0) 
    r_evals = np.sort(rho_b.eigenenergies())[::-1] 
    H_evals = np.sort(H_B.eigenenergies())         
    E_current = expect(H_B, rho_b)
    E_passive = np.sum(r_evals * H_evals)
    return E_current - E_passive

# ==========================================
# 4. محاكاة مونت كارلو الحرارية (Thermal Monte Carlo)
# ==========================================
print(f"Running Thermal Environment Simulation with n_th = {n_th}...")

plt.figure(figsize=(10, 6))

# تشغيل 3 مسارات عشوائية لرؤية القفزات في الاتجاهين
for i in range(3):
    mc_result = mcsolve(H, psi0, tlist, c_ops, [], ntraj=1)
    states_list = mc_result.states[0] if isinstance(mc_result.states[0], list) else mc_result.states
    ergo_trajectory = [calc_ergotropy(state) for state in states_list]
    plt.plot(tlist, ergo_trajectory, alpha=0.5, label=f'Thermal Jump Path {i+1}')

# حساب المتوسط الماركوفي للبيئة الحرارية
print("Calculating Markovian Average for Thermal Bath...")
me_result = mesolve(H, psi0, tlist, c_ops, [])
ergo_avg = [calc_ergotropy(state) for state in me_result.states]

plt.plot(tlist, ergo_avg, 'k--', linewidth=2.5, label='Thermal Average (Markovian)')

# ==========================================
# 5. رسم النتائج
# ==========================================
plt.title('Quantum Battery Ergotropy in a Thermal Environment ($T > 0$)', fontsize=14)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Ergotropy (Useful Energy)', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()
