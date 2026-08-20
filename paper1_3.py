import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from scipy.integrate import simpson

def run_villam_protocol(N_qubits):
    print(f"\n{'='*50}")
    print(f"🚀 Starting Quantum Battery Analysis for N = {N_qubits} Qubits")
    print(f"{'='*50}")

    # المعاملات الأساسية
    N_cavity = 4
    w_b = 1.0
    g = 0.5
    lam_decay = 0.2

    # تعريف المؤثرات في فضاء ديك المتناظر (Dicke Basis)
    Jz = tensor(jmat(N_qubits/2, 'z'), qeye(N_cavity))
    Jp = tensor(jmat(N_qubits/2, '+'), qeye(N_cavity))
    Jm = tensor(jmat(N_qubits/2, '-'), qeye(N_cavity))
    # الحجم الصحيح للمؤثر a هو N_qubits + 1
    a  = tensor(qeye(N_qubits + 1), destroy(N_cavity))
    H_B_total = Jz + N_qubits/2

    # الحالات الابتدائية
    psi1_init = tensor(basis(N_qubits+1, 0), basis(N_cavity, 0)) # مشحونة بالكامل
    rho1_init = ket2dm(psi1_init)
    psi2_init = tensor(basis(N_qubits+1, N_qubits), basis(N_cavity, 0)) # فارغة
    rho2_init = ket2dm(psi2_init)

    # تجهيز حساب الإرجوتروبي
    H_evals = np.sort(jmat(N_qubits/2, 'z').eigenenergies() + N_qubits/2)
    def calc_ergo(state):
        rho_b = state.ptrace(0)
        evals = np.sort(rho_b.eigenenergies())[::-1]
        E_curr = expect(H_B_total.ptrace(0), rho_b)
        E_pass = np.sum(evals * H_evals)
        return max(0, E_curr - E_pass)

    # ---------------------------------------------------------
    # القسم الأول: ديناميكا اللاماركوفية (BLP) والزمن
    # ---------------------------------------------------------
    tlist_dyn = np.linspace(0, 15, 150)
    
    def run_dynamics(delta, gamma0):
        w_p = w_b + delta
        H = w_b * (Jz + N_qubits/2) + w_p * a.dag() * a + g * (Jp * a + Jm * a.dag())
        def decay(t, args): return np.sqrt(args['gamma0'] * np.exp(-lam_decay * t))
        c_ops = [[a, decay]]
        
        res1 = mesolve(H, rho1_init, tlist_dyn, c_ops, [], args={'gamma0': gamma0})
        res2 = mesolve(H, rho2_init, tlist_dyn, c_ops, [], args={'gamma0': gamma0})

        ergo = [calc_ergo(s) for s in res1.states]
        dist = [tracedist(res1.states[i].ptrace(0), res2.states[i].ptrace(0)) for i in range(len(tlist_dyn))]
        blp = simpson(np.maximum(np.gradient(dist, tlist_dyn[1]-tlist_dyn[0]), 0), x=tlist_dyn)
        return ergo, dist, blp

    print("Calculating Time Dynamics & Trace Distance...")
    ergo_0, dist_0, blp_0 = run_dynamics(delta=0.0, gamma0=1.0)
    ergo_opt, dist_opt, blp_opt = run_dynamics(delta=0.8, gamma0=1.0)

    # الرسم البياني الأول
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(f'Dynamics for N={N_qubits} Collective Qubits', fontsize=16)
    
    ax1.plot(tlist_dyn, dist_0, 'k--', label=fr'No Filter ($\Delta=0$), BLP={blp_0:.3f}')
    ax1.plot(tlist_dyn, dist_opt, 'b-', linewidth=2, label=fr'Optimal Filter ($\Delta=0.8$), BLP={blp_opt:.3f}')
    ax1.set_ylabel(r'$D(\rho_1, \rho_2)$', fontsize=12)
    ax1.legend(); ax1.grid(alpha=0.3)
    
    ax2.plot(tlist_dyn, ergo_0, 'k--', label='Ergotropy (No Filter)')
    ax2.plot(tlist_dyn, ergo_opt, 'r-', linewidth=2, label='Ergotropy (Optimal Filter)')
    ax2.set_xlabel('Time', fontsize=12); ax2.set_ylabel('Ergotropy', fontsize=12)
    ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # القسم الثاني: الخريطة الحرارية الشاملة (Heatmap)
    # ---------------------------------------------------------
    print("Generating Parameter Space Heatmap (This will take a moment)...")
    tlist_hm = np.linspace(0, 15, 30) # نقاط زمنية أقل لتسريع المسح
    delta_vals = np.linspace(0.0, 2.0, 20)
    gamma_vals = np.linspace(0.1, 1.5, 20)
    ergo_map = np.zeros((len(gamma_vals), len(delta_vals)))

    for i, gamma0 in enumerate(gamma_vals):
        def decay_hm(t, args): return np.sqrt(args['gamma0'] * np.exp(-lam_decay * t))
        c_ops = [[a, decay_hm]]
        
        for j, delta in enumerate(delta_vals):
            w_p = w_b + delta
            H = w_b * (Jz + N_qubits/2) + w_p * a.dag() * a + g * (Jp * a + Jm * a.dag())
            res = mesolve(H, rho1_init, tlist_hm, c_ops, [], args={'gamma0': gamma0})
            ergo_map[i, j] = calc_ergo(res.states[-1])

    # الرسم البياني الثاني (الخريطة)
    plt.figure(figsize=(10, 7))
    X, Y = np.meshgrid(delta_vals, gamma_vals)
    cp = plt.contourf(X, Y, ergo_map, levels=30, cmap='magma')
    plt.colorbar(cp, label=f'Residual Ergotropy at $t=15$')
    
    # رسم الحد النظري
    theo_opt = g * np.sqrt(N_qubits / (2 * gamma_vals))
    plt.plot(theo_opt, gamma_vals, 'w--', linewidth=2.5, alpha=0.9, label=r'Theoretical $\Delta^*$')
    
    plt.title(f'Ergotropy Survival Map (N={N_qubits} Qubits)', fontsize=14)
    plt.xlabel(r'Detuning Filter ($\Delta$)', fontsize=12)
    plt.ylabel(r'Initial Decay Strength ($\gamma_0$)', fontsize=12)
    plt.xlim(0, 2.0); plt.ylim(0.1, 1.5)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
    print(f"Analysis for N={N_qubits} Complete!\n")

# ==========================================
# تشغيل البروتوكول للبطاريات المتعددة بالتتابع
# ==========================================
for N in [2, 3, 4]:
    run_villam_protocol(N)

print("🏆 All simulations finished successfully. The data is ready for Paper 1!")
