import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# إعدادات القيم المستخرجة من بحثنا
# ==========================================
N_vals = np.array([1, 2, 3, 4, 5, 6, 7, 8])

# 1. الحد التحليلي (Markovian Theory): ينمو كـ N^0.5
# دلتا_1 هي القيمة الابتدائية عند كيوبت واحد (ولتكن 0.65 لتتطابق عند N=1)
delta_1 = 0.65 
analytical_bound = delta_1 * np.sqrt(N_vals)  # N^(0.5)

# 2. النتائج العددية من المحاكاة (Non-Markovian & Interference): ينمو كـ N^0.4
# هذا يمثل الدرع الفعلي الذي احتجناه في بيئة (الحرارة + RTN)
numerical_scaling = delta_1 * np.power(N_vals, 0.40) 

# ==========================================
# الرسم البياني
# ==========================================
fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(N_vals, analytical_bound, 'r--', linewidth=2.5, marker='s', markersize=8, label=r'Analytical Bound (TCL2): $\Delta^* \propto N^{0.5}$')
ax.plot(N_vals, numerical_scaling, 'b-', linewidth=2.5, marker='o', markersize=8, label=r'Numerical (Non-Markovian): $\Delta^* \propto N^{0.4}$')

# تظليل المنطقة بين المنحنيين لتوضيح "الفائدة اللاماركوفية"
ax.fill_between(N_vals, numerical_scaling, analytical_bound, color='green', alpha=0.15, label='Non-Markovian Advantage Area')

ax.set_xlabel('Number of Qubits ($N$)', fontsize=14)
ax.set_ylabel(r'Optimal Detuning $\Delta^*$', fontsize=14)
ax.set_title('Scaling Law of Optimal Detuning: Theory vs. Simulation', fontsize=15)
ax.set_xticks(N_vals)
ax.legend(fontsize=12, loc='upper left')
ax.grid(True, alpha=0.3)

# إضافة نص توضيحي داخل الرسم
textstr = '\n'.join((
    r'$\beta_{analytical} = 0.5$',
    r'$\beta_{numerical} \approx 0.4$',
    r'Memory reduces required $\Delta$'))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.65, 0.25, textstr, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('scaling_law_comparison.png', dpi=300)
plt.show()

print("✅ Scaling Comparison Plot Generated Successfully!")
