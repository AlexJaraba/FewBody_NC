import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import subprocess
import os

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
PARAM_PATH = PROJECT_ROOT / "data" / "param.txt"
OUTPUT_PATH = PROJECT_ROOT / "output.csv"
EXECUTABLE_PATH = PROJECT_ROOT / "few_body_nc.exe"

plt.style.use('bmh')  # Use a nicer style for plots

def read_initial_conditions(filename):
    masses = []
    positions = []
    velocities = []

    with open(filename, 'r') as f:
        for line in f:
            data = line.split()
            masses.append(float(data[0]))
            positions.append([float(data[1]), float(data[2]), float(data[3])])
            velocities.append([float(data[4]), float(data[5]), float(data[6])])

    return np.array(masses), np.array(positions), np.array(velocities)

def plot_initial_conditions(positions):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for pos in positions:
        ax.scatter(pos[0], pos[1], pos[2])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect('equal')
    plt.title('Initial Positions of Bodies')
    plt.show()

def read_output(filename):
    df = pd.read_csv(filename)

    ids = sorted(df["id"].unique())

    data = {}

    for i in ids:
        body= df[df["id"] == i]
        data[i] = {
            "time": body["time"].values,
            "x": body["x"].values,
            "y": body["y"].values,
            "z": body["z"].values,
            "vx": body["vx"].values,
            "vy": body["vy"].values,
            "vz": body["vz"].values,
            "mass": body["mass"].values
        }

    return data, df

def PlotVerificationSuite(data,df):
    G = 0.000296014912

    times = sorted(df["time"].unique())

    energies = []
    angular_momentum = []
    linear_momentum = []
    com_positions = []

    for t in times:
        step = df[df["time"] == t]

        pos = step[["x","y","z"]].values
        vel = step[["vx","vy","vz"]].values
        m   = step["mass"].values

        # Kinetic Energy
        KE = 0.5 * np.sum(m * np.sum(vel**2, axis=1))

        # Potential Energy
        PE = 0.0
        N = len(m)

        for i in range(N):
            for j in range(i+1, N):
                r = np.linalg.norm(pos[i] - pos[j]) + 1e-12
                PE -= G * m[i] * m[j] / r

        # Total Energy
        E = KE + PE
        energies.append(E)

        # Angular Momentum
        L = np.zeros(3)
        for i in range(N):
            L += np.cross(pos[i], m[i] * vel[i])
        angular_momentum.append(np.linalg.norm(L))

        # Linear Momentum
        P = np.sum(m[:,None] * vel, axis=0)
        linear_momentum.append(np.linalg.norm(P))

        # Center of Mass Position
        Rcm = np.sum(m[:,None] * pos, axis=0) / np.sum(m)
        com_positions.append(np.linalg.norm(Rcm))

    energies = np.array(energies)
    angular_momentum = np.array(angular_momentum)
    linear_momentum = np.array(linear_momentum)
    com_positions = np.array(com_positions)

    # Relative errors
    E0 = energies[0]
    L0 = angular_momentum[0]
    P0 = linear_momentum[0]
    Rcm0 = com_positions[0]

    dE = np.abs(energies - E0)
    dL = np.abs(angular_momentum - L0)
    dP = np.abs(linear_momentum - P0) 
    dRcm = np.abs(com_positions - Rcm0)

    # Print summary statistics
    print("Max |dE|:", np.max(np.abs(dE)))
    print("Max |dL|:", np.max(np.abs(dL)))
    print("Max |dP|:", np.max(np.abs(dP)))
    print('Max |dRcm|:', np.max(np.abs(dRcm)))

    print("Final dE:", dE[-1])
    print("Final dL:", dL[-1])
    print("Final dP:", dP[-1])
    print("Final dRcm:", dRcm[-1])

    # Figure Layout
    fig = plt.figure(figsize=(16,10))
    gs = fig.add_gridspec(2, 4)

    # Orbit Plot
    ax_orbit = fig.add_subplot(gs[:,0:2])
    for i in sorted(data.keys()):
        x = data[i]["x"]
        y = data[i]["y"]
        ax_orbit.plot(x, y, linewidth=2, label=f'Body {i}')  # Plot the trajectory of each body
        ax_orbit.scatter(x[0], y[0], s=60, zorder=10)  # Plot the initial position of each body

    ax_orbit.set_title('Orbits of Bodies')
    ax_orbit.set_xlabel('X')
    ax_orbit.set_ylabel('Y')
    ax_orbit.set_aspect('equal')
    ax_orbit.set_facecolor('#f8f8f8')
    ax_orbit.tick_params(labelsize=10)
    ax_orbit.legend()
    ax_orbit.grid(True, which='both', alpha=0.3)

    # Energy Error
    ax1 = fig.add_subplot(gs[0,2])
    ax1.semilogy(times, np.abs(dE) + 1e-16)  # Add small value to avoid log(0)
    ax1.set_title("Relative Energy Error")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("(E - E0)/E0")
    ax1.grid(True, which='both', alpha=0.3)

    # Angular Momentum Error
    ax2 = fig.add_subplot(gs[0,3])
    ax2.semilogy(times, np.abs(dL) + 1e-16)  # Add small value to avoid log(0)
    ax2.set_title("Relative Angular Momentum Error")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("(L - L0)/L0")
    ax2.grid(True, which='both', alpha=0.3)

    # Linear Momentum Error
    ax3 = fig.add_subplot(gs[1,2])
    ax3.semilogy(times, np.abs(dP) + 1e-16)  # Add small value to avoid log(0)
    ax3.set_title("Relative Linear Momentum Error")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("|P - P0|")
    ax3.grid(True, which='both', alpha=0.3)

    # Center of Mass Drift
    ax4 = fig.add_subplot(gs[1,3])
    ax4.semilogy(times, dRcm + 1e-30)  # Add small value to avoid log(0)
    ax4.set_title("Relative Center of Mass Drift")
    ax4.set_xlabel("Time")
    ax4.set_ylabel("|Rcm - Rcm0|")
    ax4.grid(True, which='both', alpha=0.3)

    fig.subplots_adjust(left=0.06, right=0.97, top=0.93, bottom=0.08, wspace=0.25, hspace=0.30)
    plt.show()

def rewrite_param(dt, runtime):
    with open(PARAM_PATH, "r") as f:
        lines = f.readlines()
    with open(PARAM_PATH, "w") as f:
        for line in lines:
            if line.startswith("timestep"):
                f.write(f"timestep {dt}\n")
            elif line.startswith("runtime"):
                f.write(f"runtime {runtime}\n")
            else:
                f.write(line)

def RunTimeStepScalingStudy():
    dt_ref = 0.00025
    dts = [0.1, 0.05, 0.025, 0.0125]
    runtime = 1000
    position_errors = []

    # -------------------------------------------------------------------
    # Run reference solution
    # -------------------------------------------------------------------

    print("\nRunning reference solution...")

    rewrite_param(dt_ref, runtime)

    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()
    
    result = subprocess.run([str(EXECUTABLE_PATH)], capture_output=True, text=True)

    print(result.stdout)

    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("Reference simulation failed")
    
    data_ref, df_ref = read_output(OUTPUT_PATH)

    # -------------------------------------------------------------------
    # Final reference positions
    # -------------------------------------------------------------------

    reference_positions = {}
    final_time_ref = max(df_ref["time"])
    final_step_ref = df_ref[df_ref["time"] == final_time_ref]

    for i in sorted(final_step_ref["id"].unique()):
        body = final_step_ref[final_step_ref["id"] == i]
        reference_positions[i] = np.array([body["x"].values[0], body["y"].values[0], body["z"].values[0]])

    # -------------------------------------------------------------------
    # Run test timesteps
    # -------------------------------------------------------------------

    for dt in dts:

        print(f"\nRunning dt = {dt}")
        rewrite_param(dt, runtime)

        if OUTPUT_PATH.exists():
            OUTPUT_PATH.unlink()
        
        result = subprocess.run([str(EXECUTABLE_PATH)], capture_output=True, text=True)

        print(result.stdout)

        if result.returncode != 0:
            print(result.stderr)
            raise RuntimeError(f"Simulation failed for dt = {dt}")
        
        data, df = read_output(OUTPUT_PATH)
        final_time = max(df["time"])
        final_step = df[df["time"] == final_time]

        # ---------------------------------------------------------------
        # Compute RMS position error
        # ---------------------------------------------------------------
        
        total_error = 0.0
        count = 0

        for i in sorted(final_step["id"].unique()):
            body = final_step[final_step["id"] == i]

            r = np.array([body["x"].values[0], body["y"].values[0], body["z"].values[0]])
            dr = r - reference_positions[i]

            total_error += np.dot(dr, dr)

            count += 1
        
        rms_error = np.sqrt(total_error / count)
        position_errors.append(rms_error)

        print(f"RMS position error: {rms_error}")
        
    # ---------------------------------------------------------------
    # Read new output
    # ---------------------------------------------------------------

    dts = np.array(dts)
    position_errors = np.array(position_errors)
    
    # ---------------------------------------------------------------
    # Plot convergence
    # ---------------------------------------------------------------

    plt.figure()
    plt.loglog(dts, position_errors, marker='o')
    plt.gca().invert_xaxis()
    plt.xlabel('Timestep (dt)')
    plt.ylabel('RMS Position Error')
    plt.title('True Convergence Test')
    plt.grid(True, which='both', ls='--')
    plt.show()

    # ---------------------------------------------------------------
    # Print convergence ratio
    # ---------------------------------------------------------------

    print("\nConvergence Ratios:")

    for i in range(len(position_errors) - 1):
        ratio = position_errors[i] / position_errors[i+1]
        print(f"{dts[i]} -> {dts[i+1]} : ratio = {ratio}")



if __name__ == "__main__":
    data, df = read_output(OUTPUT_PATH)
    PlotVerificationSuite(data, df)

    RunTimeStepScalingStudy()
