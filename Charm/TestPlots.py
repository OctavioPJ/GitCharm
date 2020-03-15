from Cube import Charm
from Cube import numpy as np
from Cube import plt
import matplotlib.animation as animation

alf = Charm()
tfinal = 3


def update(i, ax, fig):
    ax.cla()
    N = int(np.logspace(1, 5)[i])
    dt = tfinal / N
    alf.RungeKutta(_dt=dt, _N=N, order=2)

    Numeric = alf.Solution
    alf.ExactSolution(alf.t)
    line = ax.plot(alf.t, (Numeric - alf.Solution)/alf.Solution, '-', label=str(N))

    # ax.set_ylim(-5, 5)
    ax.set_xlim(-0.01, 0.5)
    ax.legend()
    ax.grid(True)
    return line,


def test_crosses_through_0():
    # [686, 868, 1098, 1389, 1757, 2222, 2811, 3556, 4498, 5689, 7196, 9102, 11513, 14563]
    for N in [3556, 4498]:
        print(N)
        dt = tfinal / N

        alf.RungeKutta(_dt=dt, _N=2 * N)
        RungeSolution = alf.Solution
        alf.ExactSolution(alf.t)
        plt.plot(alf.t, RungeSolution > alf.Solution, '-o', label='Runge Kutta Orden 2 {}'.format(N))

        alf.Euler(_dt=dt, _N=2 * N, order=1)
        EulerSolution = alf.Solution
        alf.ExactSolution(alf.t)
        plt.plot(alf.t, EulerSolution > alf.Solution, '-o', label='Euler Orden 1 {}'.format(N))

        alf.Euler(_dt=dt, _N=2 * N, order=2)
        Euler2Solution = alf.Solution
        alf.ExactSolution(alf.t)
        plt.plot(alf.t, Euler2Solution > alf.Solution, '-o', label='Euler Orden 2 {}'.format(N))

        print(alf.t[-1])
        plt.legend()
    return 0


if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # ax.set_xlim(-0.01,0.5)
    dt = tfinal / 10

    alf.RungeKutta(_dt=dt, _N=10, order=1)
    Numeric = alf.Solution
    alf.ExactSolution(alf.t)
    ax.plot(alf.t, (Numeric - alf.Solution)/alf.Solution, '-o')
    #ax.set_ylim(-6, 6)

    ax.set_xlim(-0.01, 0.5)
    ani = animation.FuncAnimation(fig, update,
                                  frames=range(50),
                                  fargs=(ax, fig), interval=100)
    plt.show()
