# MaxEnt Inverse Reinforcement Learning (IRL) for One Health-Related Tweet Popularity

## Overview

This repository contains the codebase for implementing the MaxEnt IRL approach, as proposed by Ziebart et al. (2008) and adapted for Python by Luz, M. (2019). The approach explores some factors influencing the popularity of One Health-related tweets on X (formerly Twitter), with a particular focus on the #onehealth hashtag. This repository supports the study by Requena-Mullor et al. (under review).

## Repository Contents

- **`irl_codebase.py`**: Contains the implementation of the MaxEnt IRL approach, including code for training the IRL model using expert data. The required model inputs are listed below:
- **`expert1_states_actions.csv`**: A CSV file containing the states and actions of expert #1.
- **`expert2_states_actions.csv`**: A CSV file containing the states and actions of expert #2.
- **`expert3_states_actions.csv`**: A CSV file containing the states and actions of expert #3.
- **`expert4_states_actions.csv`**: A CSV file containing the states and actions of expert #4.
- **`features_dataframe.csv`**: A CSV file containing the matrix of states' features used for training the MaxEnt IRL model.
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

Open your system's terminal or command prompt and navigate to the extracted folder:

### 3. Run the script:

python3 irl_codebase.py

### 4. Examine the Results

After running the script, the model will output the estimated weights for each feature, indicating their influence on tweet popularity (measured by retweet counts). 

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
