from Cube import Charm
from Cube import numpy as np
from Cube import plt

if __name__ == '__main__':
    tfinal = 3
    alf = Charm()
    final_euler_2 = []
    final_expon_2 = []
    final_euler_1 = []

    for n in np.logspace(1, 6):
        N = int(n)
        print(N)
        dt = tfinal / N

        alf.RungeKutta(_dt=dt, _N=N)
        final_value_2 = alf.Solution[-1]
        final_value = alf.ExactSolution(alf.t[-1])
        # final_value = alf.ExpMatrix(alf.t[-1])[0, 1]
        final_euler_2.append(abs(final_value_2 - final_value))

        alf.ExponentialMethod(_dt=dt, _N=N)
        final_value_e = alf.Solution[-1]
        final_value = alf.ExactSolution(alf.t[-1])
        # final_value = alf.ExpMatrix(alf.t[-1])[0, 1]
        final_expon_2.append(abs(final_value_e - final_value))

        alf.Euler(_dt=dt, _N=N, order=1)
        final_value_1 = alf.Solution[-1]
        final_value = alf.ExactSolution(alf.t[-1])
        # final_value = alf.ExpMatrix(alf.t[-1])[0, 1]
        final_euler_1.append(abs(final_value_1 - final_value))

    plt.loglog(np.logspace(1, 6), final_euler_2, 'o')
    plt.loglog(np.logspace(1, 6), final_euler_1, '-o')
    plt.loglog(np.logspace(1, 6), final_expon_2, 'o')