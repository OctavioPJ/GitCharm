import matplotlib.pyplot as plt
from Cube import Charm
from Cube import numpy as np

if __name__ == '__main__':

    tfinal = 3
    alf = Charm()
    final_euler = []
    final_runge = []
    final_expon = []
    final_exp = []

    for n in np.logspace(1, 5):
        N = int(n)
        print(N)
        dt = tfinal / N

        alf.Euler(_dt=dt, _N=2*N)
        final_value_euler = alf.Solution[-1]
        # final_value = alf.ExpMatrix(alf.t[-1])[0,1]
        final_value = alf.ExactSolution(alf.t[-1])
        final_euler.append(abs(final_value_euler-final_value))

        alf.RungeKutta(_dt=dt, _N=2*N)
        final_value_runge = alf.Solution[-1]
        # final_value = alf.ExpMatrix(alf.t[-1])[0,1]
        final_value = alf.ExactSolution(alf.t[-1])
        final_runge.append(abs(final_value_runge-final_value))

        alf.ExponentialMethod(_dt=dt, _N=2*N)
        final_value_expon = alf.Solution[-1]
        # final_value = alf.ExpMatrix(alf.t[-1])[0, 1]
        final_value = alf.ExactSolution(alf.t[-1])
        final_expon.append(abs(final_value_expon-final_value))

    plt.loglog(np.logspace(1, 5), final_euler, 'o')
    plt.loglog(np.logspace(1, 5), final_runge, '-o')
    plt.loglog(np.logspace(1, 5), final_expon, 'o')


