Dear Editors of SoftwareX,

Please consider the enclosed Original Software Publication, "EduLogGen: A Python package for generating and validating synthetic
educational interaction logs", for publication in SoftwareX.

Interaction logs from learning platforms are central to learning analytics
and educational data mining, but they are personal data that are difficult
to share. As a result, methods are often evaluated on a single private
dataset and rarely against a known ground truth. EduLogGen is an open-source
Python package (MIT licence) that covers the whole workflow for synthetic
educational logs: ingestion with pseudonymisation, analysis, interpretable
Markov and semi-Markov generators, validation with fidelity metrics and
privacy indicators, and benchmarking on held-out learners under a versioned
protocol. An experimental layer generates data with known ground truth:
labelled anomalies, controlled behavioural manipulations with a
manipulation check, and behavioural profiles. These let researchers score
anomaly detectors and clustering methods.

The manuscript evaluates the package on two public datasets: daily activity
of 10,143 learners of the Open University Learning Analytics Dataset,
aggregated into weekly sequences, and fine-grained, millisecond-timestamped
clickstreams from EdNet. It shows where each generator matches real data
relative to the variation between two real learner samples, and tests
memorisation against the training learners. It separates the reconstruction,
stability, and outcome association of behavioural profiles, and sets up
known-truth anomaly experiments on real and synthetic backgrounds, and
compares interpretable Markov generators with an optional GRU network. All
scripts and aggregate results are part of the repository.

The software is publicly available on GitHub
(https://github.com/anisbey1/EduLogGen), installable from PyPI
(`pip install eduloggen`), documented at https://anisbey1.github.io/EduLogGen/,
and archived on Zenodo (https://doi.org/10.5281/zenodo.23264161, all versions). It has
1,024 automated tests with 99% coverage and continuous integration on
Python 3.11-3.13.

This manuscript has not been published and is not under consideration
elsewhere. The author declares no competing interests. The use of generative
AI tools is disclosed in the manuscript.

Sincerely,

Anis Bey
Higher School of Management Sciences, Annaba, Algeria
beyanisse@gmail.com
ORCID: https://orcid.org/0000-0001-9410-0851
