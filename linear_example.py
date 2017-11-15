import yaml
from UQ_Dynamics import ClusteringMethod, Sample
from UQ_Dynamics.linear_model import ShallowWater as swe
import numpy as np
from UQ_Dynamics import UT
from scipy.sparse import diags
import os

#current_dir = os.path.dirname(os.path.abspath(__file__))
current_dir = os.getcwd()
filename = os.path.join(current_dir, 'cfg', 'linear_config.yml')

with open(filename, 'r') as ymfile:
  cfg = yaml.load(ymfile)

overlapping = cfg['overlapping']

lin_config = cfg['shallowwater']
simulate_config = cfg['simulation']
t0 = simulate_config['t0']
tf = simulate_config['tf']
Timestep = simulate_config['Timestep']
dt = (tf - t0) / Timestep
t = np.arange(t0, tf, dt)
print 'Libraries loaded, setting up linear model'

mdl = swe(lin_config)
sys = mdl._generate_system()

x0 = np.zeros(mdl.dim - 2) + 0.02
cov = diags(np.random.rand(mdl.dim - 2) * 5).toarray()
print 'Models loaded. Mean and covariance initialized'

ut = UT(x0, covariance=cov)
samples = ut.generate_samples()

print 'Samples generated. Clustering in process'

adjacency = mdl._adjacency_matrix(1e-12)
cluster_model = ClusteringMethod(affinity_data=adjacency, clustering_type=cfg['clustering']['method'])
cluster_matrix = cluster_model.compute_cluster_matrix_skc(overlapping=overlapping)

print 'Clustering completed. Generating samples from cluster matrix'
#sys_cl = mdl._generate_system_clusters(cluster_matrix)
sample_cluster, _ = ut.compute_samplespace_cluster(cluster_matrix, overlapping=overlapping)

t_low = int(Timestep / 16)
u = np.hstack((1. / (Timestep - 1) * np.arange(1, t_low),
                       1. / (Timestep - 1) * (Timestep / 16. - 1.) * np.ones(Timestep - t_low + 1)))

moment_clustered_model = []

Wm_cluster = sample_cluster.Wm
Wc_cluster = sample_cluster.Wc

moment_compare_list = []
n_times = Timestep
c = 0
for k in range(n_times):
    c += 1
    #if not c % 10:
    print 'Computing at time t = %f', (c*dt)

    moment_sample = np.array([mdl.simulate_one_step(sys, np.hstack((0, sample0, 0)), u[k])[0]
                              for sample0 in samples.samples])
    moment_sample = ut.compute_moments(Sample(samples.Wc, samples.Wm, moment_sample[:, 1:-1]), tol=1e-5)

    ut = UT(moment_sample._mean, covariance=moment_sample._covariance)
    samples = ut.generate_samples()

    moment_sample_cluster = np.array([mdl.simulate_one_step(sys, np.hstack((0, sample0, 0)), u[k])[0]
                                      for sample0 in sample_cluster.samples])
    moment_sample_cluster = ut.compute_moments_from_cluster(
      Sample(Wc_cluster, Wm_cluster, moment_sample_cluster[:, 1: -1]), cluster_matrix, overlapping=overlapping)

    ut = UT(moment_sample_cluster._mean, covariance=moment_sample_cluster._covariance)
    sample_cluster = ut.compute_samplespace_cluster(cluster_matrix, overlapping=overlapping)
    moment_compare_list.append(ut.compare_moment(moment_sample, moment_sample_cluster))

if cfg['visualization']:
    import matplotlib.pyplot as plt
    mean_compare = np.array([moment[0] for moment in moment_compare_list])
    covariance_compare = np.array([np.linalg.norm(moment[1]) for moment in moment_compare_list])
    plt.plot(t, mean_compare)
plt.show()