import numpy as np
import matplotlib.pyplot as plt

# --- 1. الثوابت الفيزيائية الأساسية ---
h_plank = 6.626e-34       # ثابت بلانك (J.s)
epsilon_0 = 8.854e-12     # السماحية الكهربائية للفراغ (F/m)

# --- 2. حساب الاستقطابية (Polarizability - alpha) تحليلياً ---
# تقريب الاستقطابية بناءً على الحجم: alpha ~ 4 * pi * epsilon_0 * r^3
r_atom = 1.0e-10          # نصف قطر الذرة التقريبي (1 أنجستروم)
r_nucleus = 7.0e-15       # نصف قطر نواة الثوريوم التقريبي (7 فيمتومتر)

alpha_atom = 4 * np.pi * epsilon_0 * (r_atom**3)
alpha_nucleus = 4 * np.pi * epsilon_0 * (r_nucleus**3)

# الترددات الأساسية (لغرض حساب الانزياح النسبي)
nu_atom = 4.3e14          # تردد ساعة بصرية نموذجية (Hz)
nu_nucleus = 2.0e15       # تردد ساعة الثوريوم (Hz)

# --- 3. نمذجة المجال الكهربائي الخارجي E(t) ---
# سنفترض وجود مجال كهربائي بيئي متذبذب (Stray E-fields + Blackbody Radiation)
time_duration = 10.0
points = 10000
t = np.linspace(0, time_duration, points)
dt = t[1] - t[0]

# توليد مجال كهربائي عشوائي (V/m) - متوسطه ليس صفراً دائماً في البيئات الحقيقية
E_field_noise = np.random.normal(100, 50, points) 

# إشارة الموجة التثاقلية الصافية (h_mu_nu) التي نحاول قياسها
gw_amplitude = 1e-15
pure_gw_signal = gw_amplitude * np.sin(2 * np.pi * 0.5 * t)

# --- 4. حساب تأثير شتارك ديناميكياً (Stark Shift Calculation) ---
# Delta nu_stark = (-1 / 2h) * alpha * E^2
delta_nu_stark_atom = (-1 / (2 * h_plank)) * alpha_atom * (E_field_noise**2)
delta_nu_stark_nucleus = (-1 / (2 * h_plank)) * alpha_nucleus * (E_field_noise**2)

# حساب التغير النسبي للتردد (Fractional Frequency Shift)
fractional_freq_atomic = pure_gw_signal + (delta_nu_stark_atom / nu_atom)
fractional_freq_nuclear = pure_gw_signal + (delta_nu_stark_nucleus / nu_nucleus)

# --- 5. تراكم الطور وتباين آلان (كما في الكود السابق) ---
phase_atomic = np.cumsum(fractional_freq_atomic) * dt
phase_nuclear = np.cumsum(fractional_freq_nuclear) * dt

taus = np.logspace(-2, 0.5, 50)
allan_dev_atomic = []
allan_dev_nuclear = []

for tau in taus:
    step = max(1, int(tau / dt))
    if step >= len(fractional_freq_atomic):
        break
    avg_atomic = [np.mean(fractional_freq_atomic[i:i+step]) for i in range(0, len(fractional_freq_atomic)-step, step)]
    avg_nuclear = [np.mean(fractional_freq_nuclear[i:i+step]) for i in range(0, len(fractional_freq_nuclear)-step, step)]
    
    if len(avg_atomic) > 1:
        allan_dev_atomic.append(np.sqrt(np.mean(np.diff(avg_atomic)**2) / 2))
        allan_dev_nuclear.append(np.sqrt(np.mean(np.diff(avg_nuclear)**2) / 2))
    else:
        allan_dev_atomic.append(np.nan)
        allan_dev_nuclear.append(np.nan)

taus = taus[:len(allan_dev_atomic)]

# --- 6. الرسم البياني ---
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
fig.suptitle('Analytical Stark Effect Dynamics in Clocks', fontsize=18, fontweight='bold', color='white')

# (أ) الرسم الأول: استجابة الطور
ax1.plot(t, phase_atomic * 1e15, label=f'Atomic Phase Drift (Stark Shift Dominated)', color='#ff6666', alpha=0.9, linewidth=1.5)
ax1.plot(t, phase_nuclear * 1e15, label=f'Nuclear Phase (Stark Shift Negligible)', color='#66b3ff', linewidth=2.5)
ax1.set_title('Calculated Phase Accumulation under Dynamic E-Field Noise', fontsize=14)
ax1.set_xlabel('Time (s)', fontsize=12)
ax1.set_ylabel('Phase Error (x $10^{-15}$)', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.3)
ax1.legend(loc='upper left', fontsize=11)


# (ب) الرسم الثاني: تباين آلان
ax2.loglog(taus, allan_dev_atomic, label='Atomic Stability (Limited by E-field Variance)', color='#ff6666', linewidth=2)
ax2.loglog(taus, allan_dev_nuclear, label='Nuclear Stability (Quantum Limited)', color='#66b3ff', linewidth=2)
ax2.set_title('Allan Deviation Derived from Stark Polarizability', fontsize=14)
ax2.set_xlabel('Averaging Time $\\tau$ (s)', fontsize=12)
ax2.set_ylabel('Fractional Instability $\sigma_y(\\tau)$', fontsize=12)
ax2.grid(True, which="both", linestyle='--', alpha=0.3)
ax2.legend(loc='upper right', fontsize=11)


plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('stark_effect_analytical.png', dpi=300)
