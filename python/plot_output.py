import matplotlib.pyplot as plt
import numpy as np

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
    import csv
    from collections import defaultdict

    data_by_body = defaultdict(lambda: {'x': [], 'y': [], 'z': []})

    with open(filename, 'r') as f:
        reader = csv.DictReader(f)

        for row in reader:
            i = int(row['id'])

            data_by_body[i]['x'].append(float(row['x']))
            data_by_body[i]['y'].append(float(row['y']))
            data_by_body[i]['z'].append(float(row['z']))
    
    x = np.array([data_by_body[i]['x'] for i in sorted(data_by_body)])
    y = np.array([data_by_body[i]['y'] for i in sorted(data_by_body)])
    z = np.array([data_by_body[i]['z'] for i in sorted(data_by_body)])

    print("Shape:", x.shape)

    return x, y, z

def plot_output(x,y,z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    nbodies = x.shape[0]

    for i in range(nbodies):
        ax.plot(x[i], y[i], z[i])  # Plot the trajectory of each body

        if len(x[i]) > 0:
            ax.scatter(x[i][0], y[i][0], z[i][0], marker='o')  # Plot the trajectory of each body

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1,1,1])

    plt.title('Orbits of Bodies')
    plt.show()

if __name__ == "__main__":
    x, y, z = read_output('output.csv')
    plot_output(x,y,z)