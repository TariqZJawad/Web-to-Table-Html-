import numpy as np
import matplotlib.pyplot as plt
from qutip import *

print("🚀 Starting Environment B Unified Analysis (Thermal + Effective RTN)...")

# ==========================================
# 1. إعدادات النظام للبيئة الواقعية
# ==========================================
N_vals = [2, 3, 4]
N_cavity = 6
w_b = 1.0
g = 0.5
n_th = 0.15          # متوسط الفوتونات الحرارية في المختبر (Thermal Noise)
gamma_phi = 0.05     # الأثر الفعال لضجيج RTN (Pure Dephasing)
t_final = 10.0       # زمن المحاكاة
tlist = np.linspace(0, t_final, 50)

# نطاقات الخرائط الحرارية (Survival Maps)
gamma_vals = np.linspace(0.1, 1.0, 12) # مسح قوة الارتباط الحراري
delta_vals = np.linspace(0.0, 2.0, 15) # مسح الإزاحة

# إعداد اللوحة الشاملة (3 صفوف × عمودين)
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
plt.subplots_adjust(hspace=0.4, wspace=0.3)

for i, N in enumerate(N_vals):
    print(f"🔄 Processing N = {N} Qubits...")
    
    # تعريف المؤثرات وتأمين الأبعاد (Qobj)
    iden_b = qeye(N + 1)
    iden_c = qeye(N_cavity)
    
    Jz = tensor(jmat(N/2, 'z'), iden_c)
    Jp = tensor(jmat(N/2, '+'), iden_c)
    Jm = tensor(jmat(N/2, '-'), iden_c)
    a  = tensor(iden_b, destroy(N_cavity))
    
    # هاميلتونيان الكيوبتات فقط (لحساب الإرجوتروبي)
    H_b_only = jmat(N/2, 'z') + (N/2) * qeye(N+1)
    H_evals = np.sort(H_b_only.eigenenergies())

    def calc_ergo(state):
        rho_b = state.ptrace(0)
        evals = np.sort(rho_b.eigenenergies())[::-1]
        E_curr = expect(H_b_only, rho_b)
        E_pass = np.sum(evals * H_evals)
        return max(0, E_curr - E_pass)

    psi_init = tensor(basis(N+1, 0), basis(N_cavity, 0))
    rho_init = ket2dm(psi_init)

    # ==========================================
    # الجزء الأول: الديناميكا (العمود الأيسر)
    # ==========================================
    gamma_test = 0.5
    delta_opt_test = g * np.sqrt(N / (2 * gamma_test)) + 0.2
    
    # مؤثرات الانهيار للبيئة الواقعية (حرارة + ضجيج طوري)
    c_ops_dyn = [
        np.sqrt(gamma_test * (1 + n_th)) * a,  # انبعاث حراري
        np.sqrt(gamma_test * n_th) * a.dag(),  # امتصاص حراري
        np.sqrt(gamma_phi) * Jz                # ضجيج RTN المدمج
    ]
    
    # بدون فلتر
    H_0 = w_b*Jz + (N/2)*w_b*tensor(iden_b, iden_c) + w_b*a.dag()*a + g*(Jp*a + Jm*a.dag())
    res_0 = mesolve(H_0, rho_init, tlist, c_ops_dyn, [])
    ergo_0 = [calc_ergo(s) for s in res_0.states]
    
    # مع الفلتر المثالي
    H_opt = w_b*Jz + (N/2)*w_b*tensor(iden_b, iden_c) + (w_b+delta_opt_test)*a.dag()*a + g*(Jp*a + Jm*a.dag())
    res_opt = mesolve(H_opt, rho_init, tlist, c_ops_dyn, [])
    ergo_opt = [calc_ergo(s) for s in res_opt.states]
    
    ax_dyn = axes[i, 0]
    ax_dyn.plot(tlist, ergo_0, 'k--', linewidth=2, label=r'$\Delta=0$ (No Filter)')
    ax_dyn.plot(tlist, ergo_opt, 'r-', linewidth=2.5, label=fr'$\Delta={delta_opt_test:.2f}$ (Opt Filter)')
    ax_dyn.set_title(f'Dynamics N={N} (Thermal + RTN)', fontsize=13)
    ax_dyn.set_xlabel('Time')
    ax_dyn.set_ylabel('Ergotropy')
    ax_dyn.legend()
    ax_dyn.grid(True, alpha=0.3)

    # ==========================================
    # الجزء الثاني: الخريطة الحرارية (العمود الأيمن)
    # ==========================================
    ergo_map = np.zeros((len(gamma_vals), len(delta_vals)))
    
    for r, gam in enumerate(gamma_vals):
        c_ops_map = [
            np.sqrt(gam * (1 + n_th)) * a,
            np.sqrt(gam * n_th) * a.dag(),
            np.sqrt(gamma_phi) * Jz
        ]
        for c, delt in enumerate(delta_vals):
            H_map = w_b*Jz + (N/2)*w_b*tensor(iden_b, iden_c) + (w_b+delt)*a.dag()*a + g*(Jp*a + Jm*a.dag())
            # نحسب فقط اللحظة الأخيرة لتوفير الوقت
            res_map = mesolve(H_map, rho_init, [0, t_final], c_ops_map, [])
            ergo_map[r, c] = calc_ergo(res_map.states[-1])
            
    ax_map = axes[i, 1]
    X, Y = np.meshgrid(delta_vals, gamma_vals)
    cmap = ax_map.contourf(X, Y, ergo_map, levels=25, cmap='inferno')
    fig.colorbar(cmap, ax=ax_map, label='Residual Ergotropy')
    
    # رسم الحد النظري
    theo_delta = g * np.sqrt(N / (2 * gamma_vals)) + 0.2
    ax_map.plot(theo_delta, gamma_vals, 'w--', linewidth=2.5, label=r'Theoretical $\Delta^*$')
    ax_map.set_xlim(0, max(delta_vals))
    ax_map.set_title(f'Survival Map N={N}', fontsize=13)
    ax_map.set_xlabel(r'Detuning Filter $\Delta$')
    ax_map.set_ylabel(r'Thermal Coupling $\gamma_{th}$')
    ax_map.legend(loc='upper right', fontsize=10)

fig.suptitle('Environment B: Robustness Against Thermal & Phase Noise (N=2 to 4)', fontsize=18, fontweight='bold')
plt.savefig('env_B_unified.png', dpi=300)
plt.show()

print("✅ Environment B Analysis Complete! Check the 'env_B_unified.png' file.")
