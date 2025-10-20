# MaxEnt Inverse Reinforcement Learning (IRL) for One Health-Related Tweet Popularity

## Overview

This repository contains the Python code and data used in the study "An inverse reinforcement learning approach to model health-related information popularity on X (Twitter)", published by Requena-Mullor et al. in Computing. The study investigates factors influencing the popularity of One Health-related tweets on X (formerly Twitter), with a particular focus on the #onehealth hashtag. The analysis is based on the Maximum Entropy Inverse Reinforcement Learning (MaxEnt IRL) framework, originally proposed by Ziebart et al. (2008) and adapted for Python by Luz, M. (2019).

This release has been published on Zenodo: [![DOI](https://zenodo.org/badge/965107839.svg)](https://doi.org/10.5281/zenodo.15256415).

## Repository Contents

- **`irl_codebase.py`**: Contains the implementation of the MaxEnt IRL approach, including code for training the IRL model using data from expert X accounts. The required model inputs are listed below:
- **`expert1_states_actions.csv`**: A CSV file containing the states and actions of expert #1.
- **`expert2_states_actions.csv`**: A CSV file containing the states and actions of expert #2.
- **`expert3_states_actions.csv`**: A CSV file containing the states and actions of expert #3.
- **`expert4_states_actions.csv`**: A CSV file containing the states and actions of expert #4.
- **`features_dataframe.csv`**: A CSV file containing the matrix of states' features used for training the MaxEnt IRL model (i.e., Weekly tweet count, Length of tweets, URL, Hashtag, Mention, COVID-19 cases, COVID-19 deaths, International day, and Day of the week).
- **`probability_transition_matrix.npy`**: A NumPy file containing the probability transition matrix used in the MaxEnt IRL model.

Additionally, the experts' tweets and their attributes are included:
- **`Experts_tweets.csv`**: A CSV file including:
  - Tweet ID
  - Text
  - Likes
  - Retweets

## Requirements

To run the code in this repository, you will need the following Python libraries (tested with the versions listed below):

**numpy** (version 1.24.2)

**pandas** (version 1.5.1)

**scikit-learn** (version 1.4.1.post1)

**itertools** (part of the Python standard library; tested with Python 3.10.12)

The core Python modules required to implement the MaxEnt IRL model —gridworld.py, trajectory.py, optimizer.py, and solver.py— are included in this repository. These modules were originally developed by Luz, M. (2019) and are available as part of the *irl-maxent* package available at https://github.com/qzed/irl-maxent.
Copyright (c) 2019 Maximilian Luz.

## Usage

### 1. Download and Extract the Repository

Download the repository and extract its content on your local machine.

### 2. Run the Script

Open your system's terminal or command prompt and navigate to the extracted folder.

### 3. Run the script:

python3 irl_codebase.py

### 4. Examine the Results

After running the script, the software will print the estimated weights for each feature, indicating their influence on tweet popularity (measured by retweet counts) and both the Value Function and the Optimal Policy. 

## License

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this repository and associated files to deal
in the code and data without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the code and to permit persons to whom the repository is furnished to do so, subject to the following conditions:

This permission notice shall be included in all copies or substantial portions of the repository.

THE CODE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE CODE OR THE USE OR OTHER DEALINGS IN THE
CODE.

## Acknowledgments

Luz, M. (2019). Maximum Entropy Inverse Reinforcement Learning - An Implementation. Software version 1.2.0. M.I.T. https://github.com/qzed/irl-maxent

Ziebart, B.D., Maas, A.L., Bagnell, J.A, & Dey, A.K. (2008). Maximum entropy inverse
reinforcement learning. In AAAI, pages 1433–1438
