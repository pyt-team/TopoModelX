ICML 2023 Topological Deep Learning Challenge
=====================
Welcome to the ICML 2023 Topological Deep Learning Challenge, hosted by the second annual `Topology and Geometry (TAG) in Machine Learning Workshop <https://www.tagds.com/events/conference-workshops/tag-ml23>`_ at ICML. 

Lead organizers: Mathilde Papillon and Nina Miolane (UC Santa Barbara).


Description of the Challenge
-------------

The purpose of this challenge is to foster reproducible reasearch in Topological Deep Learning, by crowdsourcing the open-source implementation of neural networks on topological domains. Participants are asked to contribute code for a previously existing Topological Neural Network (TNN), and test it on a toy dataset. 

Implementations and are built using TopoModelX, a Python package for deep learning on topological domains. Each submission takes the form of a Python script defining a layer of a given TNN, and a Jupyter Notebook implementing and testing a TNN built with this layer. The TNN layer leverages the coding infrastructure and building blocks from the package `TopoModelX <https://github.com/pyt-team/TopoModelX/tree/main/topomodelx>`_. Participants submit their Python script and Jupyter Notebook via `Pull Requests <https://github.com/pyt-team/TopoModelX/pulls>`_ to this GitHub repository, see Guidelines below.

Every submission respecting the submission requirements will be included in a white paper summarizing findings of the challenge. Participants will have the opportunity to co-author this publication.

**Note:** *We invite participants to review this file regularly, as details are added to the guidelines when questions are submitted to the organizers.*

Deadline
-------------
The final Pull Request submission date and time must take place before **July 13, 2023 at 16:59 (Pacific Standard Time)**.
Participants are welcome to modify their Pull Request until this time.

Winners announcement
-------------
The first three winners will be announced at the ICML 2023 TAG in Machine Learning Workshop and advertised through the web. Winners will also be contacted directly via email.

How to Submit
-------------
Anyone can participate and participation is free. It is sufficient to:

- send a Pull Request
- respect Submission Requirements.

An acceptable Pull Request automatically subscribes a participant to the challenge.

Guidelines
-------------
We encourage the participants to start submitting their Pull Request early on. This allows to debug the tests and helps to address potential issues with the code.

Teams are accepted and there is no restriction on the number of team members.

There is no restriction on the amount of submisisons per participant/team.

The principal developpers of TopoModelX are not allowed to participate.

Submission Requirements
-------------
The submisison must implement a pre-exisitng model from the literature included in Fig. 11 of the review `Architectures of Topological Deep Learning: A Survey of Topological Neural Networks <https://arxiv.org/pdf/2304.10031.pdf>`_.
The Pull Request contains two new files:
1. **{name of model}_layer.py** (ex.: hsn_layer.py) :

- stored in the directory topomodelx/nn/{domain of model}/, where {domain of model} is simplicial, cellular, or hypergraph.
- contains one class, {Name of model}Layer (ex.: HSNLayer), which uses TopoModelX computational primitives (e.g. message passing class Conv or Att, and potentially Aggregation class), to implement one layer of the model. One layer is equivalent to the message passing depicted in the tensor diagram representation fo the model (Fig. 11, Architectures of Topological Deep Learning).

2. **{name of model}_train.ipynb** (ex.: hsn_train.ipynb) :

- stored in the directory tutorials/
- contains the following steps:
a. 
b.
2. 
