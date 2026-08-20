import numpy as np
import matplotlib.pyplot as plt

# --- 1. إعداد المعلمات الزمنية ---
time_duration = 10.0  # مدة المحاكاة بالثواني
points = 10000        # دقة أخذ العينات
t = np.linspace(0, time_duration, points)
dt = t[1] - t[0]

# --- 2. النمذجة الرياضية للاضطرابات ---
# (أ) إشارة الموجة التثاقلية الصافية (h_mu_nu) - تأثير حقيقي نريد قياسه
gw_frequency = 0.5 # هرتز
gw_amplitude = 1e-15 # سعة التغير النسبي للتردد
pure_gw_signal = gw_amplitude * np.sin(2 * np.pi * gw_frequency * t)

# (ب) نمذجة التشويش الكهرومغناطيسي (H_Noise) بناءً على الاستقطابية (Polarizability)
# الساعة الذرية: تشويش عالي بسبب السحابة الإلكترونية (حجم كبير)
noise_level_atomic = 8e-15 
noise_atomic = np.random.normal(0, noise_level_atomic, points)

# الساعة النووية: تشويش شبه معدوم بسبب القوة النووية (حجم صغير جداً)
noise_level_nuclear = 1e-17
noise_nuclear = np.random.normal(0, noise_level_nuclear, points)

# --- 3. حساب الاستجابة الديناميكية ---
# التغير الكلي في التردد النسبي (Delta nu / nu)
fractional_freq_atomic = pure_gw_signal + noise_atomic
fractional_freq_nuclear = pure_gw_signal + noise_nuclear

# تراكم الطور (Phase Accumulation) - تكامل التغير في التردد عبر الزمن
# Delta Phi = 2 * pi * int(Delta nu) dt
phase_atomic = np.cumsum(fractional_freq_atomic) * dt
phase_nuclear = np.cumsum(fractional_freq_nuclear) * dt

# --- 4. حساب استقرار التردد (محاكاة مبسطة لتباين آلان Allan Deviation) ---
# هذا هو المعيار الأساسي في الفيزياء لتقييم استقرارية المذبذبات
taus = np.logspace(-2, 0.5, 50) # فترات القياس (Averaging times)
allan_dev_atomic = []
allan_dev_nuclear = []

for tau in taus:
    step = max(1, int(tau / dt))
    if step >= len(fractional_freq_atomic):
        break
    # حساب التباين للبيانات المجمعة
    avg_atomic = [np.mean(fractional_freq_atomic[i:i+step]) for i in range(0, len(fractional_freq_atomic)-step, step)]
    avg_nuclear = [np.mean(fractional_freq_nuclear[i:i+step]) for i in range(0, len(fractional_freq_nuclear)-step, step)]
    
    # حساب الانحراف المعياري التقريبي
    if len(avg_atomic) > 1:
        allan_dev_atomic.append(np.sqrt(np.mean(np.diff(avg_atomic)**2) / 2))
        allan_dev_nuclear.append(np.sqrt(np.mean(np.diff(avg_nuclear)**2) / 2))
    else:
        allan_dev_atomic.append(np.nan)
        allan_dev_nuclear.append(np.nan)

taus = taus[:len(allan_dev_atomic)]

# --- 5. إنشاء الرسم البياني التحليلي الشامل ---
# اختيار خلفية داكنة تناسب العروض الأكاديمية والمحاكاة المتقدمة
plt.style.use('dark_background') 
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
fig.suptitle('Analytical Study: Spacetime Perturbation Response\nAtomic vs. Nuclear Clock', fontsize=18, fontweight='bold', color='white')

# (أ) الرسم الأول: تراكم الطور 
ax1.plot(t, phase_atomic * 1e15, label='Atomic Clock (High EM Noise Coupling)', color='#ff6666', alpha=0.8, linewidth=1.5)
ax1.plot(t, phase_nuclear * 1e15, label='Nuclear Clock (Pure Gravity Signal)', color='#66b3ff', linewidth=2.5)
ax1.set_title('Phase Shift Accumulation: $\Delta \Phi(t) = \int (\Delta\\nu/\\nu) dt$', fontsize=14)
ax1.set_xlabel('Time (s)', fontsize=12)
ax1.set_ylabel('Accumulated Phase Error (Arbitrary Units x $10^{-15}$)', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.3)
ax1.legend(loc='upper left', fontsize=11)
# إضافة معادلة الهاميلتونيان لتفسير الرسم

# (ب) الرسم الثاني: تباين آلان 
ax2.loglog(taus, allan_dev_atomic, label='Atomic Oscillator Stability', color='#ff6666', linewidth=2)
ax2.loglog(taus, allan_dev_nuclear, label='Nuclear Oscillator Stability', color='#66b3ff', linewidth=2)
ax2.set_title('Frequency Stability Analysis (Simulated Allan Deviation $\sigma_y(\\tau)$)', fontsize=14)
ax2.set_xlabel('Averaging Time $\\tau$ (s)', fontsize=12)
ax2.set_ylabel('Fractional Frequency Instability $\sigma_y(\\tau)$', fontsize=12)
ax2.grid(True, which="both", linestyle='--', alpha=0.3)
ax2.legend(loc='upper right', fontsize=11)

# إضافة الخلاصة التحليلية كصندوق نصي داخل الرسم البياني

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('analytical_study_clocks.png', dpi=300, transparent=False)
