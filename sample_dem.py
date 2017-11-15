import os
import grass.script as gscript
import grass.script.setup as gsetup
import grass.script.array as garray
import numpy as np
from sklearn.preprocessing import normalize
from UQ_Dynamics import ClusteringMethod, KLExpansion
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF
import itertools
import scipy.io as sio

# Location of dataset

grass7bin_lin = 'grass72'
gisdb = '/dart11_2/data/users/arpanmuk/Titan2D/titan2d-v4.0.0/share/titan2d_examples/dem'

# Location of the initial dataset
location = "colimafinemini"
mapset = "PERMANENT_original"
#full_location = os.path.join(gisdb, location, mapset)


grass7bin = grass7bin_lin

# Do not change this part
gisbase = '/lib64/grass72'
os.environ['GISBASE'] = gisbase
os.environ['PATH'] += os.pathsep + os.path.join(gisbase, 'extrabin')
home = os.path.expanduser("~")
os.environ['PATH'] += os.pathsep + os.path.join(home, '.grass7', 'addons', 'scripts')

os.environ['GISDBASE'] = gisdb

gsetup.init(gisbase, gisdb, location, mapset)
maps = gscript.list_strings(type='rast')

# Loading map
X = garray.array(maps[0])
X = np.asarray(X)
region = gscript.region()
print 'Map Loaded'

X0 = X

dx = 10
N = region['rows']/dx
Nxy = N*N

X = X[::dx, :]
X = X[:, ::dx]

map_data = X
Y = X

#x = np.linspace(region['w'], region['e'],N)
#y = np.linspace(region['s'], region['n'],N)
#XX, YY = np.meshgrid(x, y)
#X = np.c_[XX.reshape(Nxy,1), YY.reshape(Nxy,1)]
#Y = Y.reshape(Nxy, 1)

#kernel = 1 * RBF(length_scale=500.0, length_scale_bounds=(1, 1000))
#model = GPR(kernel=kernel,n_restarts_optimizer=10)
#model.fit(X,Y)
#print model.kernel_
#print model.log_marginal_likelihood_value_

viz = 0

'''
if viz:
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(XX, YY, X)

#Adjacency = np.zeros((Nxy, Nxy))
#a = model.kernel_.k2.length_scale
#Adjacency = model.kernel_.k2(Y)
#X = Y.reshape(Nxy)
'''

Y = Y.reshape(Nxy)
domain = np.meshgrid(Y, Y)


a = 10


def cov_fun(x,y): return np.exp(-np.abs(x - y)/a)


mdl = KLExpansion(cov_fun)
Adjacency = mdl.compute_k_matrix(domain)
print 'Adjacency Matrix Created'

if viz:
    plt.imshow(Adjacency)

A = Adjacency

A = 0.5 * (A + A.T)
A = normalize(A)
cluster_model = ClusteringMethod(affinity_data=A, clustering_type='louvain')
cluster_matrix = cluster_model.compute_cluster_matrix_skc(overlapping=0)


map = dict()
map['map_data'] = map_data
map['cluster_matrix'] = cluster_matrix
map['K'] = Adjacency

sio.savemat('data/map_data.mat', map)

print 'Clustering Completed'

region, eigs = mdl.cluster_region(map_data, cluster_matrix)




Area = np.zeros((N, N))
row = 1

for cl in cluster_matrix:
    Area = Area + cl.reshape(N, N)*row
    row = row + 1

fig = plt.figure()
plt.imshow(Area)
"""

new_mapset = "PERMANENT"
gsetup.init(gisbase, gisdb, location, new_mapset)


Y = garray.array()
Y.write(mapname='colima', overwrite=True)
"""
