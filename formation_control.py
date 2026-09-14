import numpy as np
import networkx as nx
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

N = 20
edge_probability = 0.5

dt = 0.05
tolerance = 0.001
max_steps = 2000


while True:
    np.random.seed(42)

    G = nx.erdos_renyi_graph(N, edge_probability)

    if nx.is_connected(G):
        break


print("Nodes:", N)
print("No.of edges:", G.number_of_edges())
print("Is Graph connected? ", nx.is_connected(G))


positions = np.random.uniform(
    low=-8,
    high=8,
    size=(N, 2)
)


plt.ion()

fig, ax = plt.subplots(figsize=(8, 8))

writer = FFMpegWriter(
    fps=20,
    metadata={"title": "PRAJITH Formation Control"},
    bitrate=3000
)

ax.scatter(
    positions[:, 0],
    positions[:, 1],
    s=50
)

for i, j in G.edges():

    ax.plot(
        [positions[i, 0], positions[j, 0]],
        [positions[i, 1], positions[j, 1]]
    )

for i in range(N):

    ax.text(
        positions[i, 0] + 0.15,
        positions[i, 1] + 0.15,
        str(i + 1)
    )

ax.axhline(0, linewidth=0.8)
ax.axvline(0, linewidth=0.8)

ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)

ax.set_xlabel("x")
ax.set_ylabel("y")

ax.set_title("Initial Communication Graph")

ax.grid(True)
ax.set_aspect("equal")


r_P = np.array([
    [-2, -4],
    [-2, -3],
    [-2, -2],
    [-2, -1],
    [-2,  0],
    [-2,  1],
    [-2,  2],
    [-2,  3],
    [-2,  4],
    [-1,  4],
    [ 0,  4],
    [ 1,  4],
    [ 2,  4],
    [ 2,  3],
    [ 2,  2],
    [ 2,  1],
    [-1,  0],
    [ 0,  0],
    [ 1,  0],
    [ 2,  0]
])


r_R = np.array([
    [-2, -3],
    [-2, -2],
    [-2, -1],
    [-2,  0],
    [-2,  1],
    [-2,  2],
    [-2,  3],
    [-1,  3],
    [ 0,  3],
    [ 1,  3],
    [ 2,  3],
    [ 2,  2],
    [ 2,  1],
    [-1,  0],
    [ 0,  0],
    [ 1,  0],
    [ 2,  0],
    [ 0, -1],
    [ 1, -2],
    [ 2, -3]
])


r_A = np.array([
    [ 0,  3],
    [-1,  2],
    [-2,  1],
    [-3,  0],
    [-4, -1],
    [-5, -2],
    [-6, -3],
    [-7, -4],
    [ 1,  2],
    [ 2,  1],
    [ 3,  0],
    [ 4, -1],
    [ 5, -2],
    [ 6, -3],
    [ 7, -4],
    [-2, -1],
    [-1, -1],
    [ 0, -1],
    [ 1, -1],
    [ 2, -1]
])


r_J = np.array([
    [-3,  4],
    [-2,  4],
    [-1,  4],
    [ 0,  4],
    [ 1,  4],
    [ 2,  4],
    [ 3,  4],
    [ 3,  3],
    [ 3,  2],
    [ 3,  1],
    [ 3,  0],
    [ 3, -1],
    [ 3, -2],
    [ 3, -3],
    [ 2, -4],
    [ 1, -4],
    [ 0, -4],
    [-1, -4],
    [-2, -3],
    [-2, -2]
])


r_I = np.array([
    [-3,  4],
    [-2,  4],
    [-1,  4],
    [ 0,  4],
    [ 1,  4],
    [ 2,  4],
    [ 3,  4],
    [ 0,  3],
    [ 0,  2],
    [ 0,  1],
    [ 0,  0],
    [ 0, -1],
    [ 0, -2],
    [-3, -3],
    [-2, -3],
    [-1, -3],
    [ 0, -3],
    [ 1, -3],
    [ 2, -3],
    [ 3, -3]
])


r_T = np.array([
    [-3,  4],
    [-2.5,4],
    [-2,  4],
    [-1.5,4],
    [-1,  4],
    [ 0,  4],
    [ 1,  4],
    [1.5, 4],
    [ 2,  4],
    [2.5,4],
    [ 3,  4],
    [ 0,  3],
    [ 0,  2],
    [ 0,  1],
    [ 0,  0],
    [ 0, -1],
    [ 0, -2],
    [ 0, -3],
    [ 0, -4],
    [ 0, -5]
])


r_H = np.array([
    [-3,  4],
    [-3,  3],
    [-3,  2],
    [ 3,  4],
    [ 3,  3],
    [ 3,  2],
    [-3,  1],
    [-2,  1],
    [-1,  1],
    [ 0,  1],
    [ 1,  1],
    [ 2,  1],
    [ 3,  1],
    [ 3,  1],
    [-3,  0],
    [-3, -1],
    [-3, -2],
    [ 3,  0],
    [ 3, -1],
    [ 3, -2]
])


def formation_control(positions, r, G, dt, tolerance, max_steps, ax, letter, writer):

    for step in range(max_steps):
        
        u = np.zeros((N, 2))

        for i in range(N):
            for j in G.neighbors(i):
                u[i] += (positions[j] - positions[i]) - (r[j] - r[i])


        positions = positions + dt*u

        error = 0.0

        for i, j in G.edges():

            actual = positions[j] - positions[i]
            desired = r[j] - r[i]
            error += np.linalg.norm(actual - desired)

        error = error / G.number_of_edges() 

        ax.clear()

        ax.scatter(
            positions[:, 0],
            positions[:, 1],
            s=100
        )

        for i in range(N):

            ax.text(
                positions[i, 0] + 0.15,
                positions[i, 1] + 0.15,
                str(i + 1)
            )


        ax.axhline(0, linewidth=0.8)
        ax.axvline(0, linewidth=0.8)

        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)

        ax.set_xlabel("x")
        ax.set_ylabel("y")

        ax.set_title(
            f"Formation Control → {letter} "
            f"| Step = {step} "
            f"| Error = {error:.4f}"
        )

        ax.grid(True)
        ax.set_aspect("equal")

        writer.grab_frame()

        plt.pause(0.05)


        if error < tolerance:

            print(
                f"{letter} achieved "
                f"(iterations = {step + 1}, "
                f"error = {error:.4f})"
            )

            break


    return positions




letters = [
    ("P", r_P),
    ("R", r_R),
    ("A", r_A),
    ("J", r_J),
    ("I", r_I),
    ("T", r_T),
    ("H", r_H)
]


with writer.saving(
    fig,
    "PRAJITH_formation_control.mp4",
    dpi=150
):

    writer.grab_frame()

    plt.pause(3)

    for letter, r in letters:

        plt.pause(3)

        print(f"\nMoving to {letter}...")

        positions = formation_control(
            positions,
            r,
            G,
            dt,
            tolerance,
            max_steps,
            ax,
            letter,
            writer
        )


plt.ioff()
plt.show()

print("\nVideo saved as: PRAJITH_formation_control.mp4")