import numpy as np
import os

DATA_DIR = "data"
DEFAULT_FILE = os.path.join(DATA_DIR, "initial_conditions.txt")

def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)

G=0.000296014912

def write_conditions(masses, positions, velocities, filename=DEFAULT_FILE):
    ensure_data_dir()

    with open(filename, 'w') as f:
        for i in range(len(masses)):
            f.write(f"{masses[i]} "
                    f"{positions[i,0]} {positions[i,1]} {positions[i,2]} "
                    f"{velocities[i,0]} {velocities[i,1]} {velocities[i,2]}\n")
    print(f"Wrote {len(masses)} bodies to {filename}")

def write_nbody_random(n, filename=DEFAULT_FILE):
    masses = np.random.uniform(0.1, 1.0, n)
    positions = np.random.uniform(-1.0, 1.0, (n, 3))
    velocities = np.random.uniform(-0.5, 0.5, (n, 3))

    # Remove center-of-mass motion
    total_mass = np.sum(masses)
    v_cm = np.sum(masses[:, None] * velocities, axis=0) / total_mass
    velocities -= v_cm

    write_conditions(masses, positions, velocities, filename)

def write_nbody_orbits(n, filename=DEFAULT_FILE):
    masses = np.ones(n)
    masses[0] = 10.0  # central mass

    positions = np.zeros((n, 3))
    velocities = np.zeros((n, 3))

    for i in range(1, n):
        r = np.random.uniform(0.5, 5.0)
        theta = np.random.uniform(0, 2*np.pi)

        positions[i] = [r*np.cos(theta), r*np.sin(theta), 0]

        v = np.sqrt(G * masses[0] / r)
        velocities[i] = [-v*np.sin(theta), v*np.cos(theta), 0]

    # center-of-mass correction
    total_mass = np.sum(masses)
    v_cm = np.sum(masses[:, None] * velocities, axis=0) / total_mass
    velocities -= v_cm

    write_conditions(masses, positions, velocities, filename)

def write_manual_conditions(bodies, filename=DEFAULT_FILE):
    """
    bodies = [
        (mass, [x,y,z], [vx,vy,vz]),
        ...
    ]
    """
    ensure_data_dir()

    with open(filename, 'w') as f:
        for mass, pos, vel in bodies:
            f.write(f"{mass} "
                    f"{pos[0]} {pos[1]} {pos[2]} "
                    f"{vel[0]} {vel[1]} {vel[2]}\n")

    print(f"Wrote {len(bodies)} manual bodies → {filename}")
    
if __name__ == '__main__':
    write_nbody_orbits(7)
