import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# إعدادات النشر العلمي
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'lines.linewidth': 2.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# البيانات العددية المستخرجة من البيئة أ
N_vals = np.array([1, 2, 3, 4])
delta_num = np.array([0.55, 0.70, 0.81, 0.91])

# تعريف دالة التوسع (Scaling Function) لاستخراج بيتا
# شكل الدالة: C * N^beta
def scaling_law(N, C, beta):
    return C * (N ** beta)

# المطابقة (Curve Fitting) للبيانات العددية
popt_num, _ = curve_fit(scaling_law, N_vals, delta_num, p0=[0.55, 0.5])
C_num, beta_num = popt_num

# المنحنى النظري (بيتا = 0.5)
# نفترض أن الثابت C هو نفسه للتوضيح
C_th = C_num 
delta_th = scaling_law(N_vals, C_th, 0.5)

# توليد نقاط ناعمة للرسم البياني
N_smooth = np.linspace(1, 4, 100)
delta_num_fit = scaling_law(N_smooth, C_num, beta_num)
delta_th_fit = scaling_law(N_smooth, C_th, 0.5)

# الرسم
plt.figure(figsize=(7, 5))

# رسم البيانات العددية والمطابقة
plt.plot(N_vals, delta_num, 'ro', markersize=8, label='Numerical Data')
plt.plot(N_smooth, delta_num_fit, 'r-', label=fr'Numerical Fit ($\beta \approx {beta_num:.2f}$)')

# رسم المنحنى النظري
plt.plot(N_smooth, delta_th_fit, 'k--', label=r'Theoretical Bound ($\beta = 0.5$)')

plt.xlabel('Number of Qubits ($N$)')
plt.ylabel(r'Optimal Detuning $\Delta^*$')
plt.title(r'Scaling Law: Analytical vs Numerical $\Delta^*(N)$')
plt.xticks(N_vals)
plt.legend()
plt.grid(alpha=0.3, linestyle='--')

plt.savefig('Figure_12_Scaling_Law.png')
plt.close()

print(f"تم الحفظ بنجاح! قيمة بيتا العددية المستخرجة: {beta_num:.3f}")
