import numpy as np
from numpy import *


def factors(x):
    assert len(x) == 11
    phi = dict()
    phi['a'] = np.array([x[0], 1 - x[0]])
    phi['ab'] = np.array([[x[1], 1 - x[1]],
                          [x[2], 1 - x[2]]])
    phi['ae'] = np.array([[x[3], 1 - x[3]],
                          [x[4], 1 - x[4]]])
    phi['bc'] = np.array([[x[5], 1 - x[5]],
                          [x[6], 1 - x[6]]])
    phi['ced'] = np.array([[[x[7], 1 - x[7]],
                            [x[8], 1 - x[8]]],
                           [[x[9], 1 - x[9]],
                            [x[10], 1 - x[10]]]])
    return phi


def initial_clique_potentials(phi):
    psi = dict()
    # TODO ...
    # convert all the factors into (2*2*2)
    psi['abe'] = (tile(phi['a'], (2,2,1)).T) * (tile(phi['ab'], (2,1,1)).transpose((1,2,0))) * (tile(phi['ae'], (2,1,1)).transpose((1,0,2))) # a*b*e
    psi['bce'] = tile(phi['bc'], (2,1,1)).transpose((1,2,0)) #b*c*e
    psi['cde'] = phi['ced'].transpose((0,2,1)) #c*d*e
    # temp = np.zeros((2, 2, 2))
    # temp[:, :, 0] = phi['a'][0] * np.dot(phi['ab'][0].reshape((2, 1)), phi['ae'][0].reshape((1, 2)))
    # temp[:, :, 1] = phi['a'][1] * np.dot(phi['ab'][1].reshape((2, 1)), phi['ae'][1].reshape((1, 2)))
    # psi['abe'] = temp # b*e*a(2*2*2)
    # a = temp[:, :, 0] + temp[:, :, 1] #p(be)
    # temp[:, :, 0] = np.dot(a[0].reshape((2, 1)), phi['bc'][0].reshape((1, 2)))
    # temp[:, :, 1] = np.dot(a[1].reshape((2, 1)), phi['bc'][1].reshape((1, 2)))
    # psi['bce'] = temp # e*c*b(2*2*2)
    # a = temp[:, :, 0] + temp[:, :, 1] #p(ce)
    # temp[:,:,0] = a[:,0].reshape((2,1)) * phi['ced'][:,:,0]
    # temp[:,:,1] = a[:,1].reshaoe((2,1)) * phi['ced'][:,:,1]
    # psi['cde'] = temp # e*d*c(2*2*2)
    return psi


def messages(psi):
    delta = dict()
    # TODO ...
    delta['be_bce'] = np.sum(psi['abe'], axis=0) #b*e
    delta['ce_cde'] = np.sum(tile(delta['be_bce'], (2,1,1)).transpose((1,0,2)) * psi['bce'], axis=0) #c*e
    delta['ce_bce'] = np.sum(psi['cde'], axis=2) #c*e
    delta['be_abe'] = np.sum(tile(delta['ce_bce'], (2,1,1)) * psi['bce'], axis=2) #b*e
    return delta


def beliefs(psi, delta):
    beta, mu = dict(), dict()
    # TODO ...
    return beta, mu


def query1(beta, mu):
    # TODO ...
    return q1


def query2(beta, mu):
    # TODO ...
    return q2


def query3(beta, mu):
    # TODO ...
    return q3


def belief_propagation(phi):
    psi = initial_clique_potentials(phi)
    delta = messages(psi)
    beta, mu = beliefs(psi, delta)
    return beta, mu


def main():
    # Try BP for a given set of parameters
    x = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99])
    beta, mu = belief_propagation(factors(x))
    print(query1(beta, mu))
    print(query2(beta, mu))
    print(query3(beta, mu))


if __name__ == '__main__':
    main()
