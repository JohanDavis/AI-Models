import numpy as np

# Features: [Pclass, Sex (male=1, female=0), Age]
# Targets: Survived (1) or not (0)
# ~50 samples taken from the attached CSV (hard-coded)

x = np.array([
	[3, 1, 22], [1, 0, 38], [3, 0, 26], [1, 0, 35], [3, 1, 35],
	[1, 1, 54], [3, 1, 2], [3, 0, 27], [2, 0, 14], [3, 0, 4],
	[1, 0, 58], [3, 1, 20], [3, 1, 39], [3, 0, 14], [2, 0, 55],
	[3, 1, 2], [3, 0, 31], [2, 1, 35], [2, 1, 34], [3, 0, 15],
	[1, 1, 28], [3, 0, 8], [3, 0, 38], [1, 1, 19], [1, 1, 40],
	[2, 1, 66], [1, 1, 28], [1, 1, 42], [3, 1, 21], [3, 0, 18],
	[3, 0, 14], [3, 0, 40], [2, 0, 27], [2, 0, 3], [3, 0, 19],
	[3, 0, 18], [3, 1, 7], [3, 1, 21], [1, 0, 49], [2, 0, 29],
	[1, 1, 65], [2, 0, 21], [3, 1, 28.5], [2, 0, 5], [3, 1, 11],
	[3, 1, 22], [1, 0, 38], [1, 1, 45], [3, 1, 4], [2, 0, 29]
], dtype=float)

y = np.array([
	0, 1, 1, 1, 0, 0, 0, 1, 1, 1,
	1, 0, 0, 0, 1, 0, 0, 0, 1, 1,
	1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
	1, 0, 0, 1, 1, 0, 0, 0, 1, 1,
	0, 1, 0, 1, 0, 0, 0, 1, 0, 1
], dtype=float)

w = np.zeros(np.shape(x)[1])
b = 0

def new_w(w, b, x, y, alpha):
    diff = 1/(1 + np.exp(-np.dot(x, w) - b)) - y
    derivative = np.dot(x.T, diff) / len(x)
    return w - alpha * derivative

def new_b(w, b, x, y, alpha):
    diff = 1/(1 + np.exp(-np.dot(x, w) - b)) - y
    derivative = np.sum(diff) / len(x)
    return b - alpha * derivative

alpha = 0.01

# Feature scaling
mean = np.mean(x, axis=0)
std = np.std(x, axis=0)
x = (x - mean) / std

iteration = 0

while True:
    iteration += 1
    newW = new_w(w, b, x, y, alpha)
    newB = new_b(w, b, x, y, alpha)
	
    print(iteration, newW, newB)

    if np.max(np.abs(w - newW)) < 1e-7 and abs(b - newB) < 1e-7:
        w = newW
        b = newB
        break

    w = newW
    b = newB

# Interactive prediction (collect inputs and predict survived/not)
print()
print('Training finished. Use the prompt to predict survival for a passenger.')
while True:
	try:
		pclass = int(input('Enter Pclass (1, 2 or 3): ').strip())
		sex_raw = input('Enter sex (male/female or 1/0): ').strip().lower()
		if sex_raw in ('male', 'm'):
			sex = 1
		elif sex_raw in ('female', 'f'):
			sex = 0
		else:
			sex = int(sex_raw)
		age_input = input('Enter age (leave blank for 0): ').strip()
		age = float(age_input) if age_input != '' else 0.0
	except Exception:
		print('Invalid input — please try again.')
		continue

	features_raw = np.array([pclass, sex, age], dtype=float)
	features = (features_raw - mean) / std
	prob = 1.0 / (1.0 + np.exp(- (np.dot(features, w) + b)))
	survived = prob >= 0.5

	print(f'Predicted survival probability: {prob:.3f}')
	print('Predicted:', 'Survived (1)' if survived else 'Not survived (0)')
    
	print()
    
	again = input('Predict another passenger? (y/N): ').strip().lower()
	if again != 'y':
		break
	print()

