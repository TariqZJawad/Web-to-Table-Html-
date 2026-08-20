import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# ==========================================
# 1. المعاملات الأساسية للبطارية والدرع
# ==========================================
N = 5
w_b = 1.0
delta_filter = 0.3
w_p = w_b + delta_filter
g = 0.5

# معاملات البيئة الديناميكية (الليل والنهار)
args = {
    'gamma': 0.2,
    'n_max': 10,           # ذروة الحرارة في "الظهيرة الكمية"
    'w_env': np.pi / 10.0   # التردد: بحيث نحصل على "نهار" و"ليل" خلال 20 ثانية
}

sm = tensor(sigmam(), qeye(N))
a = tensor(qeye(2), destroy(N))
adag = tensor(qeye(2), create(N))

H = w_b * sm.dag() * sm + w_p * a.dag() * a + g * (sm.dag() * a + sm * a.dag())
H_B = w_b * sigmam().dag() * sigmam()
psi0 = tensor(basis(2, 0), basis(N, 0))

# ==========================================
# 2. دوال "الليل والنهار" للانهيار الحراري
# ==========================================
# دالة فقدان الطاقة (تشتد في النهار وتهدأ في الليل)
def emit_func(t, args):
    n_th = args['n_max'] * np.sin(args['w_env'] * t)**2
    return np.sqrt(args['gamma'] * (1 + n_th))

# دالة ضخ الفوضى الحرارية (تعمل في النهار وتنطفئ في الليل)
def absorb_func(t, args):
    n_th = args['n_max'] * np.sin(args['w_env'] * t)**2
    return np.sqrt(args['gamma'] * n_th)

# دمج المؤثرات مع الدوال الزمنية (Time-Dependent Collapse Operators)
c_ops = [
    [a, emit_func],
    [adag, absorb_func]
]

tlist = np.linspace(0, 20, 500)

# ==========================================
# 3. دالة الشغل الصافي (Ergotropy)
# ==========================================
def calc_ergotropy(state):
    rho_b = state.ptrace(0)
    r_evals = np.sort(rho_b.eigenenergies())[::-1]
    H_evals = np.sort(H_B.eigenenergies())
    E_current = expect(H_B, rho_b)
    E_passive = np.sum(r_evals * H_evals)
    return E_current - E_passive

# ==========================================
# 4. المحاكاة الديناميكية
# ==========================================
print("Simulating Quantum Day and Night Cycle...")

plt.figure(figsize=(10, 6))

for i in range(3):
    # لاحظ أننا أضفنا args=args لتمرير المعاملات الزمنية للخوارزمية
    mc_result = mcsolve(H, psi0, tlist, c_ops, [], args=args, ntraj=1)
    states_list = mc_result.states[0] if isinstance(mc_result.states[0], list) else mc_result.states
    ergo_trajectory = [calc_ergotropy(state) for state in states_list]
    plt.plot(tlist, ergo_trajectory, alpha=0.6, label=f'Trajectory {i+1}')

me_result = mesolve(H, psi0, tlist, c_ops, [], args=args)
ergo_avg = [calc_ergotropy(state) for state in me_result.states]
plt.plot(tlist, ergo_avg, 'k--', linewidth=2.5, label='Average (Markovian)')

# ==========================================
# 5. رسم "شمس الحرارة" في الخلفية للتوضيح
# ==========================================
# سنرسم شكل الدالة الحرارية باللون الأحمر الباهت لنعرف متى يكون النهار ومتى يكون الليل
n_th_t = args['n_max'] * np.sin(args['w_env'] * tlist)**2
plt.plot(tlist, n_th_t / args['n_max'], 'r:', alpha=0.4, linewidth=2, label='Thermal Sun (Heat Intensity)')
plt.fill_between(tlist, 0, n_th_t / args['n_max'], color='red', alpha=0.05)

plt.title('Quantum Battery Protection under "Day & Night" Thermal Cycles', fontsize=14)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Ergotropy / Heat Intensity', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()
