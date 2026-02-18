import numpy as np

x = np.array([750, 800, 850, 900, 950, 1000, 1050, 1100])
y = np.array([148, 163, 169, 185, 188, 205, 208, 223])

def new_w(w, b, alpha, m, x, y):
    diff = w * x + b - y
    derivative = np.dot(diff, x) / m
    return w - alpha * derivative

def new_b(w, b, alpha, m, x, y):
    diff = w * x + b - y
    derivative = np.sum(diff) / m
    return b - alpha * derivative

alpha = 1e-9
m = len(x)

w = 0
b = 0

loopCount = 0

while True:
    loopCount += 1

    newW = new_w(w, b, alpha, m, x, y)
    newB = new_b(w, b, alpha, m, x, y)

    if abs(newW - w) < 1e-11:
        break

    w = newW
    b = newB

print(loopCount, w, b)

num = int(input("Enter your house size: "))
print(f"Predicted price is: {w*num + b} K")