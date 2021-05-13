import matplotlib.pyplot as plt
from generate_data_set import get_data
import matplotlib.pyplot as plt
from GaitAnaylsisToolkit.LearningTools.Trainer import TPGMMTrainer, GMMTrainer, TPGMMTrainer_old
from GaitAnaylsisToolkit.LearningTools.Runner import GMMRunner, TPGMMRunner_old
from GaitAnaylsisToolkit.LearningTools.Runner import TPGMMRunner
from scipy import signal
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np
import matplotlib
from dtw import dtw
import numpy.polynomial.polynomial as poly





def plot_gmm(Mu, Sigma, ax=None):
    nbDrawingSeg = 35
    t = np.linspace(-np.pi, np.pi, nbDrawingSeg)
    X = []
    nb_state = len(Mu[0])
    patches = []

    for i in range(nb_state):
        w, v = np.linalg.eig(Sigma[i])
        R = np.real(v.dot(np.lib.scimath.sqrt(np.diag(w))))
        x = R.dot(np.array([np.cos(t), np.sin(t)])) + np.matlib.repmat(Mu[:, i].reshape((-1, 1)), 1, nbDrawingSeg)
        x = x.transpose().tolist()
        patches.append(Polygon(x, edgecolor='r'))
        ax.plot(Mu[0, i], Mu[1, i], 'm*', linewidth=10)

    p = PatchCollection(patches, edgecolor='k', color='green', alpha=0.8)
    ax.add_collection(p)

    return p

def get_gmm(file_name):
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 30}

    matplotlib.rc('font', **font)

    nb_states = 10

    runner = GMMRunner.GMMRunner(file_name)

    fig0, ax = plt.subplots(3,sharex=True)

    sIn = runner.get_sIn()
    tau = runner.get_tau()
    l = runner.get_length()
    motion = runner.get_motion()
    mu = runner.get_mu()
    sigma = runner.get_sigma()
    currF = runner.get_expData()

    # plot the forcing functions
    angles = get_data()
    for i in range(len(angles["Lhip"])):
        ax[0].plot(sIn, tau[1, i * l: (i + 1) * l].tolist(), color="b")
        ax[1].plot(sIn, tau[2, i * l: (i + 1) * l].tolist(), color="b")
        ax[2].plot(sIn, tau[3, i * l: (i + 1) * l].tolist(), color="b")

    ax[0].plot(sIn, currF[0].tolist(), color="y", linewidth=5)
    ax[1].plot(sIn, currF[1].tolist(), color="y", linewidth=5)
    ax[2].plot(sIn, currF[2].tolist(), color="y", linewidth=5)

    sigma0 = sigma[:, :2, :2]
    sigma1 = sigma[:, :3, :2]
    sigma2 = sigma[:, :4, :2]

    sigma1 = np.delete(sigma1, 1, axis=1)
    sigma2 = np.delete(sigma2, 1, axis=1)
    sigma2 = np.delete(sigma2, 1, axis=1)

    p = plot_gmm(Mu=np.array([mu[0,:], mu[1,:] ]), Sigma=sigma0, ax=ax[0])
    p = plot_gmm(Mu=np.array([mu[0, :], mu[2, :]]), Sigma=sigma1, ax=ax[1])
    p = plot_gmm(Mu=np.array([mu[0, :], mu[3, :]]), Sigma=sigma2, ax=ax[2])
    fig0.suptitle('Forcing Function')

    ax[2].set_xlabel('S')
    ax[0].set_ylabel('F')
    ax[1].set_ylabel('F')
    ax[2].set_ylabel('F')

    # fig0.tight_layout(pad=1.0, h_pad=0.15, w_pad=None, rect=None)
    ax[0].set_title("Left Hip")
    ax[1].set_title("Left Knee")
    ax[2].set_title("Left Ankle")

    plt.show()
