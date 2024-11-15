import matplotlib.colors as mcolors
import matplotlib.animation as ma
import matplotlib.pyplot as plt
from itertools import product
from typing import List
import pandas as pd
import numpy as np
import numba as nb
import imageio
import sys
import os
import pickle
import shutil
import warnings

from numpy import ndarray
from sympy.abc import kappa, gamma, omega

randomSeed = 10

if "ipykernel_launcher.py" in sys.argv[0]:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

new_cmap = mcolors.LinearSegmentedColormap.from_list(
    "new", plt.cm.jet(np.linspace(0, 1, 256)) * 0.85, N=256
)


@nb.njit
def colors_idx(phaseTheta):
    return np.floor(256 - phaseTheta / (2 * np.pi) * 256).astype(np.int32)


import seaborn as sns

sns.set(font_scale=1.1, rc={
    'figure.figsize': (6, 5),
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'grid.color': '#dddddd',
    'grid.linewidth': 0.5,
    "lines.linewidth": 1.5,
    'text.color': '#000000',
    'figure.titleweight': "bold",
    'xtick.color': '#000000',
    'ytick.color': '#000000'
})

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
if os.path.exists("/opt/conda/bin/ffmpeg"):
    plt.rcParams['animation.ffmpeg_path'] = "/opt/conda/bin/ffmpeg"
else:
    plt.rcParams[
        'animation.ffmpeg_path'] = r"C:\Users\Aero Planck\AppData\Local\Programs\Python\Python312\ffmpeg\bin\ffmpeg.exe"

sys.path.append("..")

# from swarmalatorlib.template import Swarmalators2D
###################################################################################
new_cmap = mcolors.LinearSegmentedColormap.from_list(
    "new", plt.cm.jet(np.linspace(0, 1, 256)) * 0.85, N=256
)
if os.path.exists("/opt/conda/bin/ffmpeg"):
    plt.rcParams['animation.ffmpeg_path'] = "/opt/conda/bin/ffmpeg"
else:
    plt.rcParams[
        'animation.ffmpeg_path'] = r"C:\Users\Aero Planck\AppData\Local\Programs\Python\Python312\ffmpeg\bin\ffmpeg.exe"


class Swarmalators:
    def __init__(self, agentsNum: int, dt: float, K: float, randomSeed: int = 100,
                 tqdm: bool = False, savePath: str = None, shotsnaps: int = 5, overWrite: bool = False) -> None:
        np.random.seed(randomSeed)
        self.positionX = np.random.random((agentsNum, 2)) * 2 - 1
        self.phaseTheta = np.random.random(agentsNum) * 2 * np.pi
        self.agentsNum = agentsNum
        self.dt = dt
        self.K = K
        self.tqdm = tqdm
        self.savePath = savePath
        self.shotsnaps = shotsnaps
        self.counts = 0
        self.temp = {}
        self.overWrite = overWrite

    def init_store(self):
        if self.savePath is None:
            self.store = None
        else:
            if os.path.exists(f"{self.savePath}/{self}.h5"):
                if self.overWrite:
                    os.remove(f"{self.savePath}/{self}.h5")
                else:
                    print(f"{self.savePath}/{self}.h5 already exists")
                    return False
            self.store = pd.HDFStore(f"{self.savePath}/{self}.h5")
            # self.store.close()
        self.append()
        return True

    def append(self):
        if self.store is not None:
            if self.counts % self.shotsnaps != 0:
                return
            self.store.append(key="positionX", value=pd.DataFrame(self.positionX))
            self.store.append(key="phaseTheta", value=pd.DataFrame(self.phaseTheta))
            # self.store.close()

    @property
    def deltaTheta(self) -> np.ndarray:
        """Phase difference between agents"""
        return self.phaseTheta - self.phaseTheta[:, np.newaxis]

    @property
    def deltaX(self) -> np.ndarray:
        """
        Spatial difference between agents

        Shape: (agentsNum, agentsNum, 2)

        Every cell = otherAgent - agentSelf !!!
        """
        return self.positionX - self.positionX[:, np.newaxis]

    @property
    def Fatt(self) -> np.ndarray:
        """Effect of phase similarity on spatial attraction"""
        pass

    @property
    def Frep(self) -> np.ndarray:
        """Effect of phase similarity on spatial repulsion"""
        pass

    @property
    def Iatt(self) -> np.ndarray:
        """Spatial attraction"""
        pass

    @property
    def Irep(self) -> np.ndarray:
        """Spatial repulsion"""
        pass

    @property
    def H(self) -> np.ndarray:
        """Phase interaction"""
        pass

    @property
    def G(self) -> np.ndarray:
        """Effect of spatial similarity on phase couplings"""
        pass

    @property
    def velocity(self) -> np.ndarray:
        """Self propulsion velocity"""
        pass

    @property
    def omega(self) -> np.ndarray:
        """Natural intrinsic frequency"""
        pass

    @staticmethod
    @nb.njit
    def _update(
            positionX: np.ndarray, phaseTheta: np.ndarray,
            velocity: np.ndarray, omega: np.ndarray,
            Iatt: np.ndarray, Irep: np.ndarray,
            Fatt: np.ndarray, Frep: np.ndarray,
            H: np.ndarray, G: np.ndarray,
            K: float, dt: float
    ):
        pass

    def update_temp(self):
        pass

    def update(self) -> None:
        self.update_temp()
        self.positionX, self.phaseTheta = self._update(
            self.positionX, self.phaseTheta,
            self.velocity, self.omega,
            self.Iatt, self.Irep,
            self.Fatt, self.Frep,
            self.H, self.G,
            self.K, self.dt
        )
        self.counts += 1

    def run(self, TNum: int):

        if not self.init_store():
            return

        if self.tqdm:
            iterRange = tqdm(range(TNum))
        else:
            iterRange = range(TNum)

        for idx in iterRange:
            self.update()
            self.append()
            self.counts = idx

        self.close()

    def plot(self) -> None:
        pass

    def close(self):
        if self.store is not None:
            self.store.close()


class Swarmalators2D(Swarmalators):
    def __init__(self, agentsNum: int, dt: float, K: float, randomSeed: int = 100,
                 tqdm: bool = False, savePath: str = None, shotsnaps: int = 5, overWrite: bool = False) -> None:
        np.random.seed(randomSeed)
        self.positionX = np.random.random((agentsNum, 2)) * 2 - 1
        self.phaseTheta = np.random.random(agentsNum) * 2 * np.pi
        self.agentsNum = agentsNum
        self.dt = dt
        self.K = K
        self.tqdm = tqdm
        self.savePath = savePath
        self.shotsnaps = shotsnaps
        self.counts = 0
        self.temp = {}
        self.overWrite = overWrite

    def plot(self) -> None:
        plt.figure(figsize=(6, 5))

        plt.scatter(self.positionX[:, 0], self.positionX[:, 1],
                    c=self.phaseTheta, cmap=new_cmap, alpha=0.8, vmin=0, vmax=2 * np.pi)

        cbar = plt.colorbar(ticks=[0, np.pi, 2 * np.pi])
        cbar.ax.set_ylim(0, 2 * np.pi)
        cbar.ax.set_yticklabels(['$0$', '$\\pi$', '$2\\pi$'])

    def update_temp(self):
        self.temp["deltaTheta"] = self.deltaTheta
        self.temp["deltaX"] = self.deltaX
        self.temp["distanceX"] = self.distance_x(self.temp["deltaX"])
        self.temp["distanceX2"] = self.temp["distanceX"].reshape(self.agentsNum, self.agentsNum, 1)

    @staticmethod
    @nb.njit
    def _update(
            positionX: np.ndarray, phaseTheta: np.ndarray,
            velocity: np.ndarray, omega: np.ndarray,
            Iatt: np.ndarray, Irep: np.ndarray,
            Fatt: np.ndarray, Frep: np.ndarray,
            H: np.ndarray, G: np.ndarray,
            K: float, dt: float
    ):
        dim = positionX.shape[0]
        pointX = velocity + np.sum(
            Iatt * Fatt.reshape((dim, dim, 1)) - Irep * Frep.reshape((dim, dim, 1)),
            axis=1
        ) / dim
        positionX += pointX * dt
        pointTheta = omega + K * np.sum(H * G, axis=1) / dim
        phaseTheta = np.mod(phaseTheta + pointTheta * dt, 2 * np.pi)
        return positionX, phaseTheta

    @staticmethod
    @nb.njit
    def distance_x_2(deltaX):
        return np.sqrt(deltaX[:, :, 0] ** 2 + deltaX[:, :, 1] ** 2).reshape(deltaX.shape[0], deltaX.shape[1], 1)

    @staticmethod
    @nb.njit
    def distance_x(deltaX):
        return np.sqrt(deltaX[:, :, 0] ** 2 + deltaX[:, :, 1] ** 2)

    # @staticmethod
    # @nb.njit
    # def _delta_theta(phaseTheta):
    #     dim = phaseTheta.shape[0]
    #     subTheta = phaseTheta - np.repeat(phaseTheta, dim).reshape(dim, dim)

    #     deltaTheta = np.zeros((dim, dim - 1))
    #     for i in np.arange(dim):
    #         deltaTheta[i, :i], deltaTheta[i, i:] = subTheta[i, :i], subTheta[i, i + 1 :]
    #     return deltaTheta

    # @staticmethod
    # @nb.njit
    # def _delta_x(positionX):
    #     dim = positionX.shape[0]
    #     subX = positionX - np.repeat(positionX, dim).reshape(dim, 2, dim).transpose(0, 2, 1)
    #     deltaX = np.zeros((dim, dim - 1, 2))
    #     for i in np.arange(dim):
    #         deltaX[i, :i], deltaX[i, i:] = subX[i, :i], subX[i, i + 1 :]
    #     return deltaX

    def div_distance_power(self, numerator: np.ndarray, power: float, dim: int = 2):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if dim == 2:
                answer = numerator / self.temp["distanceX2"] ** power
            else:
                answer = numerator / self.temp["distanceX"] ** power

        answer[np.isnan(answer) | np.isinf(answer)] = 0

        return answer

##################################################################################

class PeriodicalPotential(Swarmalators2D):
    def __init__(self, strengthLambda: float,
                 distanceD: float,  gamma: float, kappa: float, L: float, boundaryLength: float = 5,
                 agentsNum: int = 1000, dt: float = 0.01,
                 tqdm: bool = False, savePath: str = None, shotsnaps: int = 5,
                 uniform: bool = True, randomSeed: int = 10, overWrite: bool = False) -> None:

        self.speed = None
        np.random.seed(randomSeed)
        self.positionX = np.random.random((agentsNum, 2)) * boundaryLength
        self.phaseTheta = np.random.random(agentsNum) * 2 * np.pi - np.pi
        self.w = np.zeros(agentsNum)
        self.agentsNum = agentsNum
        self.dt = dt
        self.speedV0 = 3.0
        self.gamma = gamma
        self.kappa = kappa
        self.L = L
        self.distanceD = distanceD
        if uniform:
            self.ChiralMoment = np.concatenate([
                np.random.uniform(1, 3, size=agentsNum // 2),
                np.random.uniform(-3, -1, size=agentsNum // 2)
            ])
        else:
            self.ChiralMoment = np.concatenate([
                np.random.normal(loc=3, scale=0.5, size=agentsNum // 2),
                np.random.normal(loc=-3, scale=0.5, size=agentsNum // 2)
            ])
        self.uniform = uniform
        self.strengthLambda = strengthLambda
        self.tqdm = tqdm
        self.savePath = savePath
        self.temp = np.zeros(agentsNum)
        self.tempForK = np.zeros((agentsNum, agentsNum))
        self.speed = np.zeros((self.agentsNum, 2))
        self.eyeMask = ~np.eye(agentsNum, dtype=bool)
        self.shotsnaps = shotsnaps
        self.counts = 0
        self.boundaryLength = boundaryLength
        self.halfBoundaryLength = boundaryLength / 2
        self.randomSeed = randomSeed
        self.overWrite = overWrite

    @property
    def K_(self):
        return (self.distance_x(self.deltaX) <= self.distanceD) * self.eyeMask

    @property
    def K(self):
        return self.tempForK


    @property
    def deltaX(self) -> np.ndarray:
        return self._delta_x(self.positionX, self.positionX[:, np.newaxis],
                             self.boundaryLength, self.halfBoundaryLength)

    @staticmethod
    @nb.njit
    def _delta_x(positionX: np.ndarray, others: np.ndarray,
                 boundaryLength: float, halfBoundaryLength: float) -> np.ndarray:
        subX = positionX - others
        return positionX - (
                others * (-halfBoundaryLength <= subX) * (subX <= halfBoundaryLength) +
                (others - boundaryLength) * (subX < -halfBoundaryLength) +
                (others + boundaryLength) * (subX > halfBoundaryLength)
        )

    @property
    def pointThetaSpeed(self):
        return self._pointThetaSpeed(
                                self.w,
                                self.phaseTheta,
                                self.phaseTheta[:, np.newaxis],
                                self.ChiralMoment,
                                self.strengthLambda,
                                self.gamma,
                                self.K,
                                self.agentsNum,
                                self.dt
        )

    @staticmethod
    @nb.njit
    def _pointThetaSpeed(w: np.ndarray, phaseTheta: np.ndarray, other1: np.ndarray,
                    ChiralMoment: np.ndarray, strengthLambda: float, gamma: float,
                    K: np.ndarray, agentsNum: int, dt: float) -> np.ndarray:

        return (- gamma * w + ChiralMoment
                + strengthLambda * np.sum(K * np.sin(other1 - phaseTheta), axis=1)
#                + np.random.normal(0, 6.436e-12, size=agentsNum)
        ) * dt

    def append(self):
        if self.store is not None:
            if self.counts % self.shotsnaps != 0:
                return
            self.store.append(key="positionX", value=pd.DataFrame(self.positionX))
            self.store.append(key="phaseTheta", value=pd.DataFrame(self.phaseTheta))
            self.store.append(key="pointThetaSpeed", value=pd.DataFrame(self.temp))
            self.store.append(key="speed", value=pd.DataFrame(self.speed))

    def update(self):
#        self.speed[:, 0] += (-self.gamma * self.speed[:, 0] +
#                             self.kappa * np.sin(2 * np.pi * self.positionX[:, 0] / self.L) +
#                             np.random.normal(0, 6.436e-12, size=self.agentsNum))  * self.dt
#
#        self.speed[:, 1] += (-self.gamma * self.speed[:, 1] +
#                             self.kappa * (np.sin(2 * np.pi * self.positionX[:, 1] / self.L) + 0.25 * np.sin(4 * np.pi * self.positionX[:, 1] / self.L)) +
#                             np.random.normal(0, 6.436e-12, size=self.agentsNum))
#                             * self.dt

        self.speed[:, 0] += (-self.gamma * self.speed[:, 0] )  * self.dt

        self.speed[:, 1] += (-self.gamma * self.speed[:, 1] ) * self.dt

        self.positionX[:, 0] += (self.speed[:, 0] + self.speedV0 * np.cos(self.phaseTheta)) * self.dt
        self.positionX[:, 1] += (self.speed[:, 1] + self.speedV0 * np.sin(self.phaseTheta)) * self.dt
        self.positionX = np.mod(self.positionX, self.boundaryLength)
        self.tempForK = self.K_
        self.w += self.pointThetaSpeed
        self.phaseTheta += self.w * self.dt
        self.temp = self.phaseTheta
        self.phaseTheta += self.temp
        self.phaseTheta = np.mod(self.phaseTheta + np.pi, 2 * np.pi) - np.pi

    def __str__(self) -> str:

        if self.uniform:
            name = (
                f"PeriodicalPotential_uniform_{self.strengthLambda:.3f}"
                f"_{self.distanceD:.2f}_{self.gamma}_{self.kappa}_{self.L}"
            )
        else:
            name = (
                f"PeriodicalPotential_normal_{self.strengthLambda:.3f}"
                f"_{self.distanceD:.2f}_{self.gamma}_{self.kappa}_{self.L}"
            )

        return name

    def close(self):
        if self.store is not None:
            self.store.close()
