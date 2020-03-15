from Cube import Charm
from Cube import numpy as np
from Cube import plt

if __name__ == '__main__':
    tfinal = 10
    alf = Charm()

    for N in [1098, 1389]:
        print(N)
        dt = tfinal / N

        alf.RungeKutta(_dt=dt, _N=2*N)
        RungeSolution = alf.Solution
        alf.ExactSolution(alf.t)
        plt.plot(alf.t, RungeSolution > alf.Solution, '-o', label='Runge Kutta Orden 2 {}'.format(N))

        alf.Euler(_dt=dt, _N=2*N, order=1)
        EulerSolution = alf.Solution
        alf.ExactSolution(alf.t)
        plt.plot(alf.t, EulerSolution > alf.Solution, '-o', label='Euler Orden 1 {}'.format(N))

        alf.Euler(_dt=dt, _N=2 * N, order=2)
        Euler2Solution = alf.Solution
        alf.ExactSolution(alf.t)
        plt.plot(alf.t, Euler2Solution > alf.Solution, '-o', label='Euler Orden 2 {}'.format(N))

        print(alf.t[-1])
        plt.legend()
