import math

def dampen_ret(r, days, half_life=10):
    total_ret = 0
    current_r = r
    for _ in range(days):
        total_ret += current_r
        current_r *= (0.5 ** (1/half_life))
    return total_ret

daily_r = 0.01 # 1% daily
days = 30

old_ret = ((1.0 + daily_r) ** days) - 1.0
new_ret_10 = dampen_ret(daily_r, days, half_life=10)
new_ret_7 = dampen_ret(daily_r, days, half_life=7)

print(f"Daily signal: {daily_r*100}%")
print(f"Old 30-day projection (Compounded): {old_ret*100:.2f}%")
print(f"New 30-day projection (Dampened, HL=10): {new_ret_10*100:.2f}%")
print(f"New 30-day projection (Dampened, HL=7): {new_ret_7*100:.2f}%")
