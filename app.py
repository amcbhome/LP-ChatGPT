
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# --- STEP 2: Historical data ---
historical_weeks = np.arange(-8, 0)  # Weeks -8 to -1
historical_sales = np.array([95, 102, 98, 110, 105, 108, 112, 115])

hist_df = pd.DataFrame({"week": historical_weeks, "sales": historical_sales})

# --- STEP 3: Forecast (Linear Trend) ---
slope, intercept = np.polyfit(historical_weeks, historical_sales, 1)
future_weeks = np.arange(1, 5)  # Next 4 weeks
forecast = (slope * future_weeks + intercept).round().astype(int)

forecast_df = pd.DataFrame({"week": future_weeks, "forecast_demand": forecast})

# --- STEP 4: LP Setup ---
T = len(future_weeks)
D = forecast.astype(float)

c_purchase = 10.0  # $ per unit
h_hold = 0.5       # Holding cost per unit/week
p_short = 15.0     # Shortage penalty per unit
I0 = 20.0          # Initial inventory
I_max = 200.0
cash_budget = 5000.0

n_vars = 3 * T  # [x1..x4, I1..I4, s1..s4]
c = np.concatenate([np.full(T, c_purchase),
                    np.full(T, h_hold),
                    np.full(T, p_short)])

# --- STEP 5: Inventory balance constraints ---
A_eq = np.zeros((T, n_vars))
b_eq = D.copy()

for t in range(T):
    A_eq[t, t] = 1.0              # x_t
    A_eq[t, T + t] = -1.0         # I_t
    A_eq[t, 2*T + t] = -1.0       # s_t
    if t == 0:
        b_eq[t] -= I0             # Subtract known initial inventory
    else:
        A_eq[t, T + (t-1)] = 1.0  # I_{t-1}

# --- STEP 6: Inequality constraints ---
A_ub = []
b_ub = []

# Inventory capacity
for t in range(T):
    row = np.zeros(n_vars)
    row[T + t] = 1.0
    A_ub.append(row)
    b_ub.append(I_max)

# Cash budget
row = np.zeros(n_vars)
row[:T] = c_purchase
A_ub.append(row)
b_ub.append(cash_budget)

A_ub = np.vstack(A_ub)
b_ub = np.array(b_ub)

bounds = [(0, None) for _ in range(n_vars)]

# --- STEP 7: Solve LP ---
res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method='highs')
if not res.success:
    raise RuntimeError("LP failed: " + res.message)

z = res.x
x = z[:T]
I = z[T:2*T]
s = z[2*T:3*T]

# --- STEP 8: Results ---
results_df = pd.DataFrame({
    "week": future_weeks,
    "forecast_demand": D.astype(int),
    "purchase_x": np.round(x, 2),
    "end_inventory_I": np.round(I, 2),
    "shortage_s": np.round(s, 2)
})

summary = {
    "total_purchase_units": float(x.sum()),
    "total_purchase_cost": float((x * c_purchase).sum()),
    "total_holding_cost": float((I * h_hold).sum()),
    "total_shortage_cost": float((s * p_short).sum()),
    "total_cost": float((x * c_purchase).sum() + (I * h_hold).sum() + (s * p_short).sum())
}

print("\n=== Historical Sales ===")
print(hist_df)
print("\n=== Forecast ===")
print(forecast_df)
print("\n=== LP Results ===")
print(results_df)
print("\n=== Summary ===")
for k, v in summary.items():
    print(f"{k}: {v}")

# --- STEP 9: Plots ---
plt.figure(figsize=(10, 4))
plt.plot(historical_weeks, historical_sales, marker='o', label="Historical")
plt.plot(future_weeks, D, marker='o', label="Forecast")
plt.title("Sales History and Forecast")
plt.xlabel("Week")
plt.ylabel("Units")
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(10, 4))
plt.bar(future_weeks - 0.2, x, width=0.4, label="Purchases")
plt.bar(future_weeks + 0.2, I, width=0.4, label="End Inventory")
plt.title("Optimal Purchases and Inventory")
plt.xlabel("Week")
plt.ylabel("Units")
plt.grid(True)
plt.legend()
plt.show()
