from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt


class Charm(object):
    def __init__(self, derivative=True):
        self.__FirstPlot = True
        self.__ResidualsCalculated = False
        self.__RootsCalculated = False
        self.__Initialized = False
        self.Uex = None
        self.eAt = None
        self.t = None
        self.St = None
        self.Solution = None

        self.fig = self.ax = None

        self._Parameters(derivative)
        self._A_Matrix()
        self._DirectModel()

    def _Parameters(self, derivative):

        self.nuFis = 2.0150E-02
        self.Abs = 1.7890E-02

        self.Bet_ = np.array([0.0002170, 0.0014979, 0.0013778, 0.0028296, 0.0009288, 0.0003314])
        self.Bet = np.array(0.0071826)
        self.lmk_ = np.array([0.0124, 0.0306, 0.1135, 0.3071, 1.1905, 3.1748])

        self.lmk = self.Bet / np.sum(self.Bet_ / self.lmk_)

        self.inv_V = 7.78681E-07 if derivative else 0.0

        self.Dif = 1.1963E+00

        self.dx = 6.0  # combustible
        self.dy = 6.0  # combustible
        self.dz = 6.0  # combustible
        self.Dx = 24 * self.dx
        self.Dy = 15 * self.dy
        self.Dz = 28 * self.dz

        self.B2 = (np.pi / 149.88725086) ** 2 \
                  + (np.pi / (self.Dy + 4 * self.Dif)) ** 2 \
                  + (np.pi / 173.86986172) ** 2

        self.k = self.nuFis / (self.Abs + self.Dif * self.B2)
        self.ro = (self.k - 1) / self.k

        self.Lamb = self.inv_V / self.nuFis

    def _A_Matrix(self):
        self.A = np.array([[(self.ro - self.Bet) / self.Lamb, self.lmk],
                           [self.Bet / self.Lamb, -self.lmk]])
        return self.A

    def _DirectModel(self, file=None):
        if file is not None:
            self.Dk = np.loadtxt(file)
        else:
            self.Dk = np.loadtxt('..\\CAREM\\CUBEM\\Resultados_direct_173.dat')

    def Euler(self, _dt, _N):
        self.St = np.zeros((_N, 2))
        if not self.__Initialized:
            self._initial_condition()

        self.St[0, :] = self.u0

        for _n in range(1, _N):
            self.St[_n, 1] = self.St[_n - 1, 1] + \
                             (self.Bet * self.nuFis * self.St[_n - 1, 0] - self.lmk * self.St[_n - 1, 1]) * _dt
            self._Actualizar_Paso(_n, _dt)

        self.Solution = self.St[:, 0]
        self.t = np.linspace(0, _N * _dt, _N)
        return

    def ExponentialMethod(self, _dt, _N):
        self.St = np.zeros((_N, 2))
        if not self.__Initialized:
            self._initial_condition()

        self.St[0, :] = self.u0

        for _n in range(1, _N):
            self.St[_n, 1] = self.St[_n - 1, 1] * np.exp(-self.lmk * _dt) + \
                             (1 - np.exp(-self.lmk * _dt)) / self.lmk * (self.Bet * self.nuFis * self.St[_n - 1, 0])
            self._Actualizar_Paso(_n, _dt)

        self.Solution = self.St[:, 0]
        self.t = np.linspace(0, _N * _dt, _N)
        return

    def RungeKutta(self, _dt, _N):
        self.St = np.zeros((_N, 2))
        if not self.__Initialized:
            self._initial_condition()

        self.St[0, :] = self.u0

        for _n in range(1, _N):
            k1 = (self.Bet * self.nuFis * self.St[_n - 1, 0] - self.lmk * self.St[_n - 1, 1])
            k2 = (self.Bet * self.nuFis * self.St[_n - 1, 0] - self.lmk * (self.St[_n - 1, 1] + k1 * _dt))
            self.St[_n, 1] = self.St[_n - 1, 1] + (k1 + k2) / 2 * _dt
            self._Actualizar_Paso(_n, _dt)

        self.Solution = self.St[:, 0]
        self.t = np.linspace(0, _N * _dt, _N)
        return

    def _Actualizar_Paso(self, _n, _dt):
        if _n == 1:
            self.St[_n, 0] = \
                        (self.lmk * self.St[_n, 1] + self.inv_V / _dt * self.St[_n - 1, 0]) / \
                ((self.Abs + self.Dif * self.B2 - (1 - self.Bet) * self.nuFis) + self.inv_V / _dt)
        else:
            self.St[_n, 0] = \
                (self.lmk * self.St[_n, 1] + (self.inv_V / _dt)*(4/3)*self.St[_n-1, 0] - (self.inv_V / _dt)*(1/3)*self.St[_n-2, 0])/\
                            ((self.inv_V / _dt) + (self.Abs + self.Dif * self.B2 - (1 - self.Bet) * self.nuFis))

    def _initial_condition(self, n0=100):
        self.n0 = n0
        self.C0 = self.Bet * self.nuFis * self.n0 / self.lmk
        self.u0 = [self.n0, self.C0]
        self.__Initialized = True

    def _calculate_roots(self):
        self.s1 = (np.sqrt(self.Bet ** 2 + 2 * self.Bet * (self.lmk * self.Lamb - self.ro) + (
                self.lmk * self.Lamb + self.ro) ** 2) - self.Bet - self.lmk * self.Lamb + self.ro) / \
                  (2 * self.Lamb)

        self.s2 = -(np.sqrt(self.Bet ** 2 + 2 * self.Bet * (self.lmk * self.Lamb - self.ro) + (
                self.lmk * self.Lamb + self.ro) ** 2) + self.Bet + self.lmk * self.Lamb - self.ro) / \
                  (2 * self.Lamb)

        self.__RootsCalculated = True
        return self.s1, self.s2

    def _calculate_residuals(self):
        self.N1 = (self.ro / self.Lamb - self.s2) / (self.s1 - self.s2)
        self.N2 = (self.ro / self.Lamb - self.s1) / (self.s2 - self.s1)
        self.__ResidualsCalculated = True
        return self.N1, self.N2

    def ExactSolution(self, time):
        if not self.__Initialized:
            self._initial_condition()
        if not self.__RootsCalculated:
            self._calculate_roots()
        if not self.__ResidualsCalculated:
            self._calculate_residuals()

        self.Solution = self.Uex = self.n0 * (self.N1 * np.exp(self.s1 * time) + self.N2 * np.exp(self.s2 * time))
        self.t = np.array(time)
        return self.Uex

    def ExpMatrix(self, *time):
        self.Solution = self.eAt = np.array([(linalg.expm(self.A * _t)).dot(self.u0) for _t in time])
        self.t = np.array(time)
        return self.eAt

    def Plot(self, marker='o', *args, **kwargs):
        if self.__FirstPlot:
            self.fig = plt.figure()
            self.ax = self.fig.gca()
            self.__FirstPlot = False

        self.ax.plot(self.t, self.Solution, marker, *args, **kwargs)
        self.ax.set_xlabel('tiempo [seg]')
        self.ax.set_ylabel('Potencia Relativa[%]')
        self.ax.grid(True)

    pass  # Charm


if __name__ == '__main__':

    tfinal = 10
    alf = Charm()
    final_euler = []
    final_runge = []
    final_expon = []

    for n in np.logspace(1, 6):
        N = int(n)
        print(N)
        dt = tfinal / N

        alf.Euler(_dt=dt, _N=N)
        final_value_euler = alf.Solution[-1]
        final_value = alf.ExactSolution(time=alf.t[-1])
        final_euler.append(abs(final_value_euler-final_value))

        alf.RungeKutta(_dt=dt, _N=N)
        final_value_runge = alf.Solution[-1]
        final_value = alf.ExactSolution(time=alf.t[-1])
        final_runge.append(abs(final_value_runge-final_value))

        alf.ExponentialMethod(_dt=dt, _N=N)
        final_value_expon = alf.Solution[-1]
        final_value = alf.ExactSolution(time=alf.t[-1])
        final_expon.append(abs(final_value_expon-final_value))

    plt.loglog(np.logspace(1, 6), final_euler, 'o')
    plt.loglog(np.logspace(1, 6), final_runge, '-o')
    plt.loglog(np.logspace(1, 6), final_expon, 'o')
