import numpy as np
from qutip import basis, sigmaz, expect

# --- 1. بناء الهاميلتونيان الحر (طاقة النظام) ---
omega = 1.0  # تردد طاقة البطارية (نفرضها 1 للتبسيط)
# الهاميلتونيان H = (omega / 2) * sigma_z
H0 = 0.5 * omega * sigmaz()

print("Hamiltonian Matrix (H0):")
print(H0)
print("-" * 30)

# --- 2. تعريف حالات البطارية ---
# النظام ثنائي المستويات: 0 للمثار، 1 للأرضي
psi_excited = basis(2, 0)  # |e> بطارية مشحونة
psi_ground = basis(2, 1)   # |g> بطارية فارغة

# تحويل المتجهات إلى مصفوفة كثافة (rho = |psi><psi|)
rho_excited = psi_excited * psi_excited.dag()
rho_ground = psi_ground * psi_ground.dag()

# --- 3. حساب طاقة البطارية Tr(rho * H0) ---
energy_charged = expect(H0, rho_excited)
energy_empty = expect(H0, rho_ground)

print(f"Energy of a FULLY CHARGED battery: {energy_charged}")
print(f"Energy of an EMPTY battery: {energy_empty}")
