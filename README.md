# UQ_Dynamics
Uncertainty Quantification in Large Dynamical Systems: A framework for faster simulation of dynamical systems with uncertainty in state variables. Based on the idea of linearization, graph clustering, parallel computing and uncertainty propagation

Currently developing the codes in Python platform and making it more user friendly

Packages used: numpy, scipy, scikit-learn, chaospy, igraph, networkx, python-louvain

Download the content of the folder, Navigate to the folder, and run 'python nonlinear_example.py'. 

Please refer to the following papers: 
Mukherjee, A., Rai, R., Singla, P., Singh, T., & Patra, A. K. (2017). COMPARISON OF LINEARIZATION AND GRAPH CLUSTERING METHODS FOR UNCERTAINTY QUANTIFICATION OF LARGE SCALE DYNAMICAL SYSTEMS. International Journal for Uncertainty Quantification, 7(1).

Mukherjee, Arpan, et al. "Effect of DEM Uncertainty on Geophysical Mass Flow via Identification of Strongly Coupled Subsystem." International Journal for Uncertainty Quantification 9.6 (2019).

Currently supported for coupled oscillator problems such as diffusively coupled Van der Pol Oscillator and diffusively coupled Rossler Attractor

Updated clustering techniques: New techniques added: louvain method of community detection
Currently is being integrated with UQ Toolkit (http://www.sandia.gov/UQToolkit/index.html)


For further support, please email me at arpanmuk@buffalo.edu
