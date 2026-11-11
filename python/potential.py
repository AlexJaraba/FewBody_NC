import numpy as np

from pathlib import Path
from scipy.integrate import quad
from functions import read_param

PROJECT_DIR = Path(__file__).resolve().parent.parent
PARAM_PATH = PROJECT_DIR / "data" / "param.txt"
POUTPUT_FILE = PROJECT_DIR / "data" / "potential_grid.txt"
AOUTPUT_FILE = PROJECT_DIR / "data" / "acceleration_grid.txt"

params = read_param(PARAM_PATH)
G = float(params["gravitational_constant"])
M = 1.0
a = 2.0
b = 1.5
c = 1.0

xmin, xmax, nx = -5.0, 5.0, 41
ymin, ymax, ny = -5.0, 5.0, 41
zmin, zmax, nz = -5.0, 5.0, 41
# Grid setup for x, y, z coordinates
# dx = (xmax - xmin) / (nx - 1)
# dy = (ymax - ymin) / (ny - 1)
# dz = (zmax - zmin) / (nz - 1)

def density(m2):
    return (3*M / (4*np.pi*a*b*c) * (1 + m2)**(-5/2))
def psi_difference(m2):
    return (M / (2*np.pi*a*b*c)) * (1 + m2)**(-3/2)


def field_at(x, y, z):
    def integrand(tau):
        A = a**2 + tau
        B = b**2 + tau
        C = c**2 + tau

        denominator = np.sqrt(A * B * C)
        m2 = x**2 / A + y**2 / B + z**2 / C

        return psi_difference(m2) / denominator
    results, error = quad(integrand, 0, np.inf, epsabs=1e-10, epsrel=1e-10)

    return -np.pi * G * a * b * c * results

def generate_grid():
    xs = np.linspace(xmin, xmax, nx)
    ys= np.linspace(ymin, ymax, ny)
    zs = np.linspace(zmin, zmax, nz)

    phi_grid = np.zeros((nx, ny, nz))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            for k, z in enumerate(zs):
                phi_grid[i, j, k] = field_at(x, y, z)
    dphi_dx , dphi_dy , dphi_dz = np.gradient(phi_grid, xs, ys, zs, edge_order=2) # phi_i+1,j,k - phi_i-1,j,k / 2*dx etc.

    ax = -dphi_dx
    ay = -dphi_dy
    az = -dphi_dz

    with POUTPUT_FILE.open("w", encoding="utf-8") as f:
        # Write the grid dimensions to the output file
        f.write(f"{nx} {ny} {nz}\n")
        # Write the grid boundaries to the output file
        f.write(f"{xmin} {xmax} {ymin} {ymax} {zmin} {zmax}\n")
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    f.write(f"{phi_grid[i, j, k]:.17e}\n")
    with AOUTPUT_FILE.open("w", encoding="utf-8") as f:
        # Write the grid dimensions to the output file
        f.write(f"{nx} {ny} {nz}\n")
        # Write the grid boundaries to the output file
        f.write(f"{xmin} {xmax} {ymin} {ymax} {zmin} {zmax}\n")
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    f.write(f"{ax[i, j, k]:.17e} {ay[i, j, k]:.17e} {az[i, j, k]:.17e}\n")

    print("Potential grid saved:", POUTPUT_FILE)
    print("Acceleration grid saved:", AOUTPUT_FILE)

if __name__ == "__main__":
    generate_grid()