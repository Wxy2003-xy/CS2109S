def deposit(principal, interest, duration):
    return pow(1 + interest, duration) * principal




def balance(principal, interest, payout, duration):
    a = payout * (1 / interest) * (pow(1 + interest, duration) - 1)
    b = pow(1 + interest, duration) * principal
    return b - a

def new_balance(principal, gap, payout, duration):
    def f(interest):
        return deposit(principal, interest, duration) - balance(deposit(principal, interest, gap), interest, payout, duration - gap)
    return f
print(deposit(100, 0.05, 1))
print(deposit(100, 0.05, 2))

print(balance(100000, 0.01, 5000, 1))
print(balance(100000, 0.01, 5000, 2))

print(new_balance(1000, 2, 100, 2)(0.1))
print(new_balance(10000, 3, 1000, 3)(0.05))