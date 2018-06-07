import yaml
from UQ_Dynamics import ClusteringMethod, StatisticalLinearization, Sample, UT, SamplingMethodGenerator
from UQ_Dynamics.nonlinear_model import CoupledOscillatorExample
import numpy as np
from scipy.sparse import diags
import scipy.io as sio
import os, sys

# Append path to the UQ Toolkit folder
sys.path.append(os.environ['UQTK_INS'])

#Loading UQ libraries from UQ Toolkit
import PyUQTk.uqtkarray as uqtkarray
import PyUQTk.pce as uqtkpce
import PyUQTk.tools as uqtktools



#current_dir = os.path.dirname(os.path.abspath(__file__))
current_dir = os.getcwd()
filename = os.path.join(current_dir, 'cfg', 'nonlinear_config.yml')

with open(filename, 'r') as ymfile:
    cfg = yaml.load(ymfile)

overlapping = cfg['overlapping']
model_name = cfg['nonlinear']
nonlin_config = cfg['model']
num_osc = nonlin_config['num_osc']
datafile = model_name + '_data.mat'

datafile = os.path.join(current_dir, 'data', datafile)
mat = sio.loadmat(datafile)
config = dict()

if model_name == 'lorenz':
    config['sigma'] = mat['Sigma'][:num_osc]
    config['rho'] = mat['Rho'][:num_osc]
    config['beta'] = mat['Beta'][:num_osc]
    config['dimension'] = num_osc * 3
elif model_name == 'vanderpol':
    config['mu'] = mat['mu0'][:num_osc]
    config['dimension'] = num_osc * 2
elif model_name == 'rossler':
    config['a0'] = mat['a0'][:num_osc]
    config['b0'] = mat['b0'][:num_osc]
    config['c0'] = mat['c0'][:num_osc]
    config['dimension'] = num_osc * 3

epsilon = nonlin_config['epsilon']

simulate_config = cfg['simulation']
t0 = simulate_config['t0']
tf = simulate_config['tf']
dt = simulate_config['dt']
t = np.arange(t0, tf, dt)
n_times = len(t)
print 'Libraries loaded, setting up nonlinear model'

mdl = CoupledOscillatorExample.model(model_name, config, epsilon=epsilon)

x0 = np.random.rand(mdl.dimension)
cov = diags(np.random.rand(mdl.dimension) * 10).toarray()
print 'Models loaded. Mean and covariance initialized'

ut = SamplingMethodGenerator.sample(x0, covariance=cov)
samples = ut.generate_samples()

lin = StatisticalLinearization(mdl._velocity_function, x0, cov)
adjacency = lin._adjacency_matrix(1e-12)
cluster_model = ClusteringMethod(affinity_data=adjacency, clustering_type=cfg['clustering']['method'])
cluster_matrix = cluster_model.compute_cluster_matrix_skc(overlapping=overlapping)

print 'Clustering completed. Generating samples from cluster matix'
sys_cl = mdl._cluster_oscillators(cluster_matrix)
sample_cluster = ut.compute_samplespace_cluster(cluster_matrix, overlapping=overlapping)

print 'Samples generated. Preparing for simulation'
Y0 = []
c = 0

for sample0 in samples.samples:
    c += 1
    print 'Simulating Sample #%d', c
    Y0.append(mdl.simulate(sample0, t))

print 'Simulation completed for whole sample. Clustering in process'

print 'Simulating clustered samples'
Y0_cl = []
c = 0
for sample0 in sample_cluster.samples:
  c += 1
  print 'Simulating Sample #%d', c
  Y0_cl.append(mdl.simulate(sample0, t))

moment_compare_list = []

Wm_cluster = sample_cluster.Wm
Wc_cluster = sample_cluster.Wc

c = 0
for k in range(n_times):
    c += 1
    if not c % 500:
        print 'Computing moment at time t = %f', (c*dt)

    y0 = np.array([mdl.simulate(sample0, [0, dt])[-1] for sample0 in samples.samples])
    moment_sample = ut.compute_moments(Sample(samples.Wc, samples.Wm, y0))
    ut = SamplingMethodGenerator.sample(moment_sample._mean, covariance=moment_sample._covariance)
    samples = ut.generate_samples()

    moment_sample_cluster = np.array([y0[k] for y0 in Y0_cl])
    moment_sample_cluster, _ = ut.compute_moments_from_cluster(Sample(Wc_cluster, Wm_cluster, moment_sample_cluster),
                                                               cluster_matrix)

    moment_compare_list.append(ut.compare_moment(moment_sample, moment_sample_cluster))

if cfg['visualization']:
    import matplotlib.pyplot as plt
    mean_compare = np.array([moment[0] for moment in moment_compare_list])
    covariance_compare = np.array([np.linalg.norm(moment[1]) for moment in moment_compare_list])

    plt.plot(t, covariance_compare)
    plt.show()
