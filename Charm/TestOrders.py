from Cube import Charm
from Cube import numpy as np
from Cube import plt

if __name__ == '__main__':
    tfinal = 0.5
    alf = Charm()
    final_euler_2 = []
    final_expon_2 = []
    final_euler_1 = []
    _N_Vector = np.logspace(1, 5)
    for n in _N_Vector:
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

    plt.loglog(_N_Vector, final_euler_2, 'o')
    plt.loglog(_N_Vector, final_euler_1, '-o')
    plt.loglog(_N_Vector, final_expon_2, 'o')