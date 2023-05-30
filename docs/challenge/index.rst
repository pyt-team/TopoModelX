ICML 2023 Topological Deep Learning Challenge
=================================================
Welcome to the ICML 2023 Topological Deep Learning Challenge, hosted by the second annual `Topology and Geometry (TAG) in Machine Learning Workshop <https://www.tagds.com/events/conference-workshops/tag-ml23>`_ at ICML.

Lead organizers: Mathilde Papillon, Dr. Tegan Emerson, Dr. Henry Kvinge, Dr. Tim Doster, Dr. Bastian Rieck, Dr. Sophia Sanborn and Dr. Nina Miolane.


Description of the Challenge
----------------------------

The purpose of this challenge is to foster reproducible research in Topological Deep Learning by crowdsourcing the open-source implementation of neural networks on topological domains. Participants are asked to contribute code for a previously existing Topological Neural Network (TNN) and train it on a benchmark dataset.

Implementations are built using  `TopoModelX <https://github.com/pyt-team/TopoModelX/tree/main/topomodelx>`_, a Python package for deep learning on topological domains. Each submission takes the form of a  `Pull Request <https://github.com/pyt-team/TopoModelX/pulls>`_ to TopoModelX containing the necessary code for implementing a TNN from the literature. The implementation leverages the coding infrastructure and building blocks from TopoModelX.

*Note:* *We invite participants to review this file regularly, as details are added when questions are submitted to the organizers.*

⭐️ Publication Outcomes for Participants ⭐️
------------------------------------------
🎉 **Every submission respecting the submission requirements** will be included in a white paper summarizing findings of the challenge, published through the  `2023 TAG in Machine Learning Workshop <https://www.tagds.com/events/conference-workshops/tag-ml23>`_ at ICML. All participants with qualifying submissions will have the opportunity to co-author this publication.

📘 Participants with the top 8 best submissions will have the additional opportunity to co-author a software paper on TopoModelX submitted to the **Journal of Machine Learning Research**.

🏆 The best submissions in each topological domain (hypergraph, simplicial, cell, combinatorial) will receive special recognition at the ICML TAG in Machine Learning Workshop.

Deadline
--------
The final Pull Request submission date and time must take place before **July 13, 2023 at 16:59 (Pacific Standard Time)**.
Participants are welcome to modify their Pull Request until this time.

How to Submit
-------------
Everyone can participate and participation is free. It is sufficient to:

- send a Pull Request
- respect Submission Requirements

An acceptable Pull Request automatically subscribes a participant/team to the challenge.

Guidelines
----------
We encourage the participants to start submitting their Pull Request early on. This helps debug failing tests and helps address potential issues with the code.

In the case of multiple submissions of similar quality implementing the same TNN, earlier Pull Requests will be given priority consideration.

Teams are accepted and there is no restriction on the number of team members.

A Pull Request should contain no more than one model (TNN). There is no restriction on the amount of submissions (Pull Requests) per participant/team.

The principal developers of TopoModelX are not allowed to participate.

Submission Requirements
-----------------------
The submission must implement a pre-existing model from the literature included in `Fig. 11 <https://github.com/pyt-team/TopoModelX/blob/main/topomodelx.jpeg>`_ of the review `Architectures of Topological Deep Learning: A Survey of Topological Neural Networks <https://arxiv.org/pdf/2304.10031.pdf>`_.

.. figure:: https://github.com/pyt-team/TopoModelX/blob/main/topomodelx.jpeg
   :align: center
   :alt: tnns
   :width: 250px

All submitted code must comply with TopoModelX's GitHub Action workflow, successfully passing all tests, linting, and formatting (i.e. Black, isort, flake8).

The submission consists of a Pull Request to TopoModelX that contains three new files:

1. {name of model}_layer.py (ex.: hsn_layer.py) :

- stored in the directory ``topomodelx/nn/{domain of model}``, where ``{domain of model}`` is ``simplicial``, ``cell``, ``hypergraph``, or ``combinatorial``.
- contains one class, ``{Name of model}Layer`` (ex.: ``HSNLayer``), which uses TopoModelX computational primitives to implement one layer of the model. One layer is equivalent to the message passing depicted in the tensor diagram representation for the model (Fig. 11, Architectures of Topological Deep Learning).
- for examples, check out example layers in ``topomodelx/nn/simplicial``, ``topomodelx/nn/cell``, and ``topomodelx/nn/hypergraph``.

2. {name of model}_train.ipynb (ex.: hsn_train.ipynb) :

- stored in the directory ``tutorials/{domain of model}`` and contains the following steps:

  1. Pre-processing
        - imports necessary packages as well as ``{Name of model}Layer`` class
        - loads a benchmark dataset `using PyTorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html>`_ and assigns labels.
        - lifts the dataset from the graph domain to the topological domain of choice (hypergraph, simplicial complex, cell complex, combinatorial complex) using TopoNetX.

  2. Creating the neural network
        - defines a class ``{Name of model}`` (ex.: ``HSN``) that inherits from ``torch.nn.Module`` and uses ``{Name of model}Layer`` along with ``torch.Linear`` layers to create a Topological Neural Network.

  3. Training the neural network on a classification task
        - defines a simple training loop for node/edge/complex classification (depending on which features the model outputs).
        - note: submissions are not evaluated based on model performance, but rather code quality and accuracy of model implementation.
- examples are provided in tutorials/

3. test_{name_of_model}_layer.py (ex.: test_hsn_layer.py)

- stored in directory ``test/nn/{domain of model}``
- contains one class, ``Test{Name of model}Layer`` (ex.: ``TestHSNLayer``), which contains unit tests for all of the functions contained in the ``{Name of model}Layer`` class. Please use pytest (not unittest).
  - for examples, check out ``test/nn/simplicial``, ``test/nn/cell``, and ``test/nn/hypergraph``.

  **Note :** in the case that ``{Name of model}Layer`` requires further manipulation of the computational primitives in ``topomodelx/base``, a Pull Request may include modifications to the files in ``topomodelx/base`` or new files in ``topomodelx/base``. Every single new function MUST be accompanied by a new unit test stored in an appropriately named/located test file. With that being said, we highly encourage participants to make the most of TopoModelX's computational primitives as is and only resort to this option if absolutely necessary (ex.: implementing a new attention function or aggregation method).

Evaluation
----------

The `Condorcet method <https://en.wikipedia.org/wiki/Condorcet_method>`_ will be used to rank the submissions and decide on the winners in each topological domain. The evaluation criteria will be:

- Does the submission implement the chosen model correctly, specifically in terms of its message passing scheme? (The training schemes do not need to match that of the original model).
- How readable/clean is the implementation? How well does the submission respect TopoModelX's APIs?
- Is the submission well-written? Do the docstrings clearly explain the methods? Are the unit tests robust?

Note that these criteria do not reward model performance, nor complexity of training. Rather, the goal is to implement well-written and accurate model architectures that will foster reproducible research in our field.

Selected TopoModelX maintainers and collaborators, as well as each team whose submission(s) respect(s) the guidelines, will vote once on Google Form to express their preference for the best submission in each topological domain. Note that each team gets only one vote/domain, even if there are several participants in the team.

A link to a Google Form will be provided to record the votes. While the form will ask for an email address to identify the voter, voters' identities will remain secret--only the final ranking will be shared.

Questions
---------
Feel free to contact us through GitHub issues on this repository, or through the `Geometry and Topology in Machine Learning slack <https://tda-in-ml.slack.com/join/shared_invite/enQtOTIyMTIyNTYxMTM2LTA2YmQyZjVjNjgxZWYzMDUyODY5MjlhMGE3ZTI1MzE4NjI2OTY0MmUyMmQ3NGE0MTNmMzNiMTViMjM2MzE4OTc#/>`_. Alternatively, you can contact Mathilde Papillon at papillon@ucsb.edu.

