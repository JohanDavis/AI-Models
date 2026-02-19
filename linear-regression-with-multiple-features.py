import numpy as np

# Features: [house size, bedrooms, age]
x = np.array([
    [179, 2, 72], [123, 2, 34], [168, 5, 38], [237, 5, 53], [127, 4, 23],
    [ 69, 5, 69], [128, 3, 24], [130, 5, 88], [ 76, 5,  2], [ 59, 4, 79],
    [235, 5, 49], [203, 1, 94], [171, 5,  1], [111, 2, 14], [ 73, 5, 74],
    [101, 2, 46], [ 81, 5, 25], [211, 1, 51], [123, 2, 53], [ 95, 2, 84],
    [ 74, 2, 46], [118, 4, 95], [197, 1, 78], [177, 2, 55], [102, 5, 21],
    [161, 2, 25], [156, 3, 84], [179, 2,  5], [137, 5,  8], [127, 4, 59],
    [ 65, 2, 93], [103, 1, 30], [188, 4, 20], [199, 4, 41], [ 99, 2, 68],
    [119, 5, 43], [150, 1, 34], [170, 4, 45], [222, 1, 20], [187, 2, 53],
    [ 85, 3, 59], [ 70, 2, 96], [ 54, 5,  4], [172, 3, 11], [102, 3, 36],
    [ 60, 4,  5], [178, 2, 93], [ 88, 2, 57], [115, 2, 68], [ 64, 4, 88]
])

# Targets: House prices
y = np.array([
    1657., 1257., 1894., 2646., 1448.,  414., 1513., 1044., 1023.,  223.,
    2555., 1691., 2206., 1184.,  348.,  876.,  871., 2170., 1067.,  457.,
     653.,  765., 1773., 1673., 1179., 1798., 1232., 2187., 1700., 1157.,
      66., 1119., 2215., 2127.,  745., 1173., 1568., 1702., 2468., 1890.,
     675.,  141.,  760., 2051.,  952.,  764., 1429.,  713.,  913.,   96.
])

def new_w(w, b, alpha, x, y):
    # m = len(x) # no of training examples
    # n = len(x[0]) # no of features
    # new_w = np.zeros(n)

    # for j in range(n):
    #     derivative = 0
    #     for i in range(m):
    #         diff = np.dot(w, x[i]) + b - y[i]
    #         derivative += diff * x[i][j]
    #     new_w[j] = w[j] - (alpha/len(x)) * derivative
    
    # return new_w

    diff = np.dot(x, w) + b - y
    derivative = np.dot(x.T, diff) / len(x)
    return w - alpha * derivative

def new_b(w, b, alpha, x, y):
    diff = np.dot(x, w) + b - y
    derivative = np.sum(diff) / len(x)
    return b - alpha * derivative

alpha = 1e-7

w = np.zeros(x.shape[1]) # [0, 0, 0]
b = 0

loopCount = 0

while True:
    loopCount += 1

    newW = new_w(w, b, alpha, x, y)
    newB = new_b(w, b, alpha, x, y)

    if np.max(np.abs(newW - w)) < 1e-4 and abs(newB - b) < 1e-4:
        break

    w = newW
    b = newB
    print(loopCount, w, b)

print(loopCount, w, b)

while True:
    print()
    print("----------------------------------------")
    print()
    size = float(input("Enter house size in sqm: "))
    bedrooms = float(input("Enter number of bedrooms: "))
    age = float(input("Enter house age in years: "))

    features = np.array([size, bedrooms, age])
    predicted = np.dot(w, features) + b

    print(f"Predicted price is: £{predicted * 1000:,.0f}")

    again = input("Do you want to predict another house? (y): ")
    if again.lower() != "y":
        break