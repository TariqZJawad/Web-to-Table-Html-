import numpy as np
import matplotlib.pyplot as plt
from qutip import *
import time

print("🚀 Starting Final Environment B Code (Thermal + RTN)...")

# ==========================================
# 1. إعدادات النظام للبيئة الواقعية
# ==========================================
N_vals = [2, 3, 4]
N_cavity = 4          # حجم تجويف مخفض لضمان اكتمال الحسابات لـ N=4 بوقت معقول
w_b = 1.0
g = 0.5
gamma_phi = 0.05      # ضجيج إزالة الترابط الطوري (RTN Effect)
t_final = 15.0
tlist = np.linspace(0, t_final, 60)

# إعدادات موجة الحرارة
T_avg = 0.6
T_amp = 0.4
tau_cycle = 10.0

def n_th_t(t, args):
    T_t = T_avg + T_amp * np.sin(2 * np.pi * t / tau_cycle)
    T_t = max(T_t, 0.05) # منع الوصول للصفر المطلق
    return 1.0 / (np.exp(w_b / T_t) - 1.0)

# ==========================================
# 2. الحلقة الرئيسية للبطاريات
# ==========================================
for N in N_vals:
    print(f"\n=========================================")
    print(f"🔄 Processing Battery with N = {N} Qubits...")
    start_time = time.time()
    
    # تعريف المؤثرات (Qobj)
    iden_b = qeye(N + 1)
    iden_c = qeye(N_cavity)
    
    Jz = tensor(jmat(N/2, 'z'), iden_c)
    Jp = tensor(jmat(N/2, '+'), iden_c)
    Jm = tensor(jmat(N/2, '-'), iden_c)
    a  = tensor(iden_b, destroy(N_cavity))
    
    H_b_only = jmat(N/2, 'z') + (N/2) * qeye(N+1)
    H_evals = np.sort(H_b_only.eigenenergies())

    def calc_ergo(state):
        rho_b = state.ptrace(0)
        evals = np.sort(rho_b.eigenenergies())[::-1]
        E_curr = expect(H_b_only, rho_b)
        E_pass = np.sum(evals * H_evals)
        return max(0, E_curr - E_pass)

    # الحالات الابتدائية لحساب BLP (مقياس مسافة الأثر)
    psi_init_1 = tensor(basis(N+1, 0), basis(N_cavity, 0)) # بطارية مشحونة بالكامل
    rho_init_1 = ket2dm(psi_init_1)
    
    psi_init_2 = tensor(basis(N+1, N), basis(N_cavity, 0)) # بطارية فارغة تماماً
    rho_init_2 = ket2dm(psi_init_2)

    # ---------------------------------------------------------
    # الجزء الأول: الديناميكا (الحرارة + BLP + الإرجوتروبي)
    # ---------------------------------------------------------
    print(f"  > Simulating Dynamics & BLP...")
    gamma_dyn = 0.5
    delta_opt = g * np.sqrt(N / (2 * gamma_dyn)) + 0.2
    
    # دوال الانحلال المعتمدة على الزمن للحرارة
    def gamma_em(t, args): return np.sqrt(gamma_dyn * (1 + n_th_t(t, args)))
    def gamma_ab(t, args): return np.sqrt(gamma_dyn * n_th_t(t, args))
    
    c_ops_dyn = [
        [a, gamma_em],
        [a.dag(), gamma_ab],
        np.sqrt(gamma_phi) * Jz  # ضجيج RTN مستمر
    ]
    
    # المحاكاة بدون فلتر
    H_0 = w_b*Jz + (N/2)*w_b*tensor(iden_b, iden_c) + w_b*a.dag()*a + g*(Jp*a + Jm*a.dag())
    res_0_st1 = mesolve(H_0, rho_init_1, tlist, c_ops_dyn, [])
    res_0_st2 = mesolve(H_0, rho_init_2, tlist, c_ops_dyn, [])
    
    ergo_0 = [calc_ergo(s) for s in res_0_st1.states]
    blp_0 = [tracedist(s1.ptrace(0), s2.ptrace(0)) for s1, s2 in zip(res_0_st1.states, res_0_st2.states)]
    
    # المحاكاة مع الفلتر
    H_opt = w_b*Jz + (N/2)*w_b*tensor(iden_b, iden_c) + (w_b+delta_opt)*a.dag()*a + g*(Jp*a + Jm*a.dag())
    res_opt_st1 = mesolve(H_opt, rho_init_1, tlist, c_ops_dyn, [])
    res_opt_st2 = mesolve(H_opt, rho_init_2, tlist, c_ops_dyn, [])
    
    ergo_opt = [calc_ergo(s) for s in res_opt_st1.states]
    blp_opt = [tracedist(s1.ptrace(0), s2.ptrace(0)) for s1, s2 in zip(res_opt_st1.states, res_opt_st2.states)]

    # الرسم البياني للصورة الأولى
    T_plot = [T_avg + T_amp * np.sin(2 * np.pi * t / tau_cycle) for t in tlist]
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 1.5]})
    
    # الشريط العلوي (الحرارة و BLP)
    color_T = 'tab:orange'
    ax1.plot(tlist, T_plot, color=color_T, linewidth=2, label='Lab Temp $T(t)$')
    ax1.set_ylabel('Temp $T(t)$', color=color_T, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=color_T)
    ax1.set_title(f'N={N} Qubits: Thermal Cycle & Information Flow (BLP)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticklabels([])
    
    ax1_twin = ax1.twinx()
    color_blp = 'tab:blue'
    ax1_twin.plot(tlist, blp_0, 'k--', linewidth=2, label='BLP (No Filter)')
    ax1_twin.plot(tlist, blp_opt, 'b-', linewidth=2.5, label='BLP (Opt Filter)')
    ax1_twin.set_ylabel('Trace Distance $D(t)$', color=color_blp, fontsize=12)
    ax1_twin.tick_params(axis='y', labelcolor=color_blp)
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', fontsize=10)

    # الشريط السفلي (الإرجوتروبي)
    ax2.plot(tlist, ergo_0, 'k--', linewidth=2, label=r'No Filter ($\Delta=0$)')
    ax2.plot(tlist, ergo_opt, 'r-', linewidth=2.5, label=fr'Optimal Filter ($\Delta={delta_opt:.2f}$)')
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Residual Ergotropy', fontsize=12)
    ax2.legend(loc='upper right', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig1.savefig(f'N{N}_Dynamics_BLP.png', dpi=300)
    plt.close(fig1)
    
    # ---------------------------------------------------------
    # الجزء الثاني: خريطة بقاء الإرجوتروبي
    # ---------------------------------------------------------
    print(f"  > Calculating Survival Heatmap (This takes a moment)...")
    gamma_vals = np.linspace(0.1, 1.0, 10)
    delta_vals = np.linspace(0.0, 2.0, 12)
    ergo_map = np.zeros((len(gamma_vals), len(delta_vals)))
    
    n_th_static = 0.25 # متوسط الضجيج الحراري للخريطة
    
    for r, gam in enumerate(gamma_vals):
        c_ops_map = [
            np.sqrt(gam * (1 + n_th_static)) * a,
            np.sqrt(gam * n_th_static) * a.dag(),
            np.sqrt(gamma_phi) * Jz
        ]
        for c, delt in enumerate(delta_vals):
            H_map = w_b*Jz + (N/2)*w_b*tensor(iden_b, iden_c) + (w_b+delt)*a.dag()*a + g*(Jp*a + Jm*a.dag())
            res_map = mesolve(H_map, rho_init_1, [0, t_final], c_ops_map, [])
            ergo_map[r, c] = calc_ergo(res_map.states[-1])
            
    fig2, ax_map = plt.subplots(figsize=(8, 6))
    X, Y = np.meshgrid(delta_vals, gamma_vals)
    cmap = ax_map.contourf(X, Y, ergo_map, levels=25, cmap='inferno')
    fig2.colorbar(cmap, ax=ax_map, label='Residual Ergotropy')
    
    theo_delta = g * np.sqrt(N / (2 * gamma_vals)) + 0.2
    ax_map.plot(theo_delta, gamma_vals, 'w--', linewidth=2.5, label=r'Theoretical $\Delta^*$')
    ax_map.set_xlim(0, max(delta_vals))
    
    ax_map.set_title(f'N={N} Qubits: Thermal & RTN Survival Map', fontsize=14)
    ax_map.set_xlabel(r'Detuning Filter $\Delta$')
    ax_map.set_ylabel(r'Thermal Coupling $\gamma_{th}$')
    ax_map.legend(loc='upper right')
    
    plt.tight_layout()
    fig2.savefig(f'N{N}_Survival_Map.png', dpi=300)
    plt.close(fig2)
    
    print(f"✅ Finished N = {N} in {time.time() - start_time:.1f} seconds. Saved 2 images.")

print("🎉 ALL DONE! Check your folder for the 6 newly generated images.")
