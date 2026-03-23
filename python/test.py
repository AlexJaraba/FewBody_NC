import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("output.csv")
G = 0.000296014912

times = sorted(df["time"].unique())

energies = []
angular_momentum = []

for t in times:
    step = df[df["time"] == t]

    pos = step[["x","y","z"]].values
    vel = step[["vx","vy","vz"]].values
    m   = step["mass"].values

    KE = 0.5 * np.sum(m[:,None] * vel**2)
    PE = 0.0
    N = len(m)

    for i in range(N):
        for j in range(i+1, N):
            r = np.linalg.norm(pos[i] - pos[j]) + 1e-12
            PE -= G * m[i] * m[j] / r

    E = KE + PE
    energies.append(E)

    L = np.zeros(3)
    for i in range(N):
        L += np.cross(pos[i], m[i] * vel[i])

    angular_momentum.append(np.linalg.norm(L))

energies = np.array(energies)
angular_momentum = np.array(angular_momentum)

E0 = energies[0]
L0 = angular_momentum[0]

dE = (energies - E0) / abs(E0)
dL = (angular_momentum - L0) / abs(L0)

print("Max |dE|:", np.max(np.abs(dE)))
print("Max |dL|:", np.max(np.abs(dL)))

print("Final dE:", dE[-1])
print("Final dL:", dL[-1])

plt.figure()
plt.plot(times, dE)
plt.title("Relative Energy Error")
plt.xlabel("Time")
plt.ylabel("(E - E0)/E0")
plt.grid()

plt.figure()
plt.plot(times, dL)
plt.title("Relative Angular Momentum Error")
plt.xlabel("Time")
plt.ylabel("(L - L0)/L0")
plt.grid()

plt.show()