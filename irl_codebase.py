#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 13:46:30 2022

@author: Juan M Requena-Mullor

"""
import numpy as np
import pandas as pd
from itertools import product
from sklearn.preprocessing import StandardScaler

import sys, os
#sys.path.append(os.path.abspath(os.path.join('~/set_the_system_path_to_the_irl_maxent_folder')))
sys.path.append(os.path.abspath(os.getcwd()))
import gridworld as W  # basic grid-world MDPs
import trajectory as T  # trajectory generation
import optimizer as O  # stochastic gradient descent optimizer
import solver as S  # MDP solver (value-iteration)


'''
The functions below were extracted from
Luz, M. (2019). Maximum Entropy Inverse Reinforcement Learning - An Implementation. Software version 1.2.0. M.I.T. https://github.com/qzed/irl-maxent
See Luz, M. (2019) for further details.
Copyright (c) 2019 Maximilian Luz
'''

def feature_expectation_from_trajectories(features, trajectories):
    n_states, n_features = features.shape

    fe = np.zeros(n_features)

    for t in trajectories:                  # for each trajectory
        for s in t.states():                # for each state in trajectory
            print(s)
            fe += features[s, :]            # sum-up features

    return fe / len(trajectories)           # average over trajectories

# get the feature expectation
def feature_expectation_from_trajectories(features, trajectories):
    n_states, n_features = features.shape

    fe = np.zeros(n_features)

    for t in trajectories:  # for each trajectory
        for s in t.states():  # for each state in trajectory
            fe += features[s, :]  # sum-up features

    return fe / len(trajectories)  # average over trajectories


def initial_probabilities_from_trajectories(n_states, trajectories):
    p = np.zeros(n_states)

    for t in trajectories:  # for each trajectory
        p[t.transitions()[0][0]] += 1.0  # increment starting state

    return p / len(trajectories)  # normalize


def compute_expected_svf(pp_transition, p_initial, terminal, reward, eps=1e-5):
    n_states, _, n_actions = pp_transition.shape
    nonterminal = set(range(n_states)) - set(terminal)  # nonterminal states

    # Backward Pass
    # 1. initialize at terminal states
    zs = np.zeros(n_states)  # zs: state partition function
    zs[terminal] = 1.0

    # 2. perform backward pass
    for _ in range(2 * n_states):  # longest trajectory: n_states
        # reset action values to zero
        za = np.zeros((n_states, n_actions))  # za: action partition function

        # for each state-action pair
        for s_from, a in product(range(n_states), range(n_actions)):

            # sum over s_to
            for s_to in range(n_states):
                za[s_from, a] += pp_transition[s_from, s_to, a] * np.exp(reward[s_from]) * zs[s_to]

        # sum over all actions
        zs = za.sum(axis=1)

    # 3. compute local action probabilities
    p_action = za / zs[:, None]

    # Forward Pass
    # 4. initialize with starting probability
    d = np.zeros((n_states, 2 * n_states))  # d: state-visitation frequencies
    d[:, 0] = p_initial

    # 5. iterate for N steps
    for t in range(1, 2 * n_states):  # longest trajectory: n_states

        # for all states
        for s_to in range(n_states):

            # sum over nonterminal state-action pairs
            for s_from, a in product(nonterminal, range(n_actions)):
                d[s_to, t] += d[s_from, t - 1] * p_action[s_from, a] * pp_transition[s_from, s_to, a]

    # 6. sum-up frequencies
    return d.sum(axis=1)


def maxent_irl(pp_transition, features, terminal, trajectories, optim, init, eps=1e-5):# 1e-4
    n_states, _, n_actions = pp_transition.shape
    _, n_features = features.shape

    # compute feature expectation from trajectories
    e_features = feature_expectation_from_trajectories(features, trajectories)

    # compute starting-state probabilities from trajectories
    p_initial = initial_probabilities_from_trajectories(n_states, trajectories)

    # gradient descent optimization
    omega = init(n_features)  # initialize our parameters
    delta = np.inf  # initialize delta for convergence check

    optim.reset(omega)  # re-start optimizer
    while delta > eps:  # iterate until convergence
        omega_old = omega.copy()

        # compute per-state reward from features
        reward = features.dot(omega)

        # compute gradient of the log-likelihood
        e_svf = compute_expected_svf(pp_transition, p_initial, terminal, reward)
        grad = e_features - features.T.dot(e_svf)
        l.append(grad)

        # perform optimization step and compute delta for convergence
        optim.step(grad)

        # re-compute detla for convergence check
        delta = np.max(np.abs(omega_old - omega))
        #print(omega)
        ll.append(omega_old)

    # re-compute per-state reward and return
    return features.dot(omega), omega, l, ll

#######################################################################################

## import experts' trajectories
# List of filenames
filenames = [
    "expert1_states_actions.csv",
    "expert2_states_actions.csv",
    "expert3_states_actions.csv",
    "expert4_states_actions.csv"
]

# Base directory
base_path = os.path.abspath(os.getcwd())

# Create a list to store all the trajectory arrays
trajectories = []

for file in filenames:
    df = pd.read_csv(os.path.join(base_path, file))
    trajectory = [(df.iloc[i-1, 5], df.iloc[i, 6], df.iloc[i, 5]) for i in range(1, df.shape[0])]
    trajectories.append(trajectory)

arr1, arr2, arr3, arr4 = trajectories
trajectoriesT = list([T.Trajectory(arr1), T.Trajectory(arr2), T.Trajectory(arr3), T.Trajectory(arr4)])# all experts' trajectories together


## import the covariates file and create the features object
# Load features data
irl_covariates = pd.read_csv('features_dataframe.csv')

# Discard the "expert" column
irl_sub = irl_covariates.iloc[:, 1:13]

# Aggregate by state
irl_cov_aggreg = irl_sub.groupby("state").mean()

# Handle NaNs or infinite values
irl_cov_aggreg.replace([np.inf, -np.inf], np.nan, inplace=True)
irl_cov_aggreg.dropna(inplace=True)

# Preserve column order for the 9 covariates used in "features"
covariate_cols = ['weekday', 'intDay_ratio', 'deaths', 'cases',
                  'lengthOFtweet', 'numbTweetsPerWeek', 'url', 'hashtag', 'mention']

# Standardize only the selected covariates
scaler = StandardScaler()
scaled_covariates = scaler.fit_transform(irl_cov_aggreg[covariate_cols])

# Create a DataFrame with the same index and standardized values
scaled_df = pd.DataFrame(scaled_covariates, index=irl_cov_aggreg.index, columns=covariate_cols)

# Manually select rows in your original order: [4,1,2,0,3]
features = np.array([
    scaled_df.iloc[4].values,
    scaled_df.iloc[1].values,
    scaled_df.iloc[2].values,
    scaled_df.iloc[0].values,
    scaled_df.iloc[3].values
])


## import transition probabilities
pp_transition = np.load('probability_transition_matrix.npy')


## set model parameters
terminal = [4]# the terminal state is cell 4th, actually there are 5 states: 0, 1, 2, 3, 4

# choose our parameter initialization strategy:
#   initialize parameters with constant
init = O.Constant(1.0)


# choose our optimization strategy:
#   we select exponentiated stochastic gradient descent with linear learning-rate decay; Kivinen et. al. 1997
optim = O.Sga(lr=O.linear_decay(lr0=0.001,decay_rate=0.07, decay_steps=1))# this parameter undirectley controls the range of weights

# actually do some inverse reinforcement learning
l = list()
ll = list()
reward_maxent = maxent_irl(pp_transition, features, terminal, trajectoriesT, optim, init)# np.transpose(np.array([features[:,0]]))


##############################################################################################################################
###########------------    Optimal Value Function and Policy        -----------------------###################################

import gridworld as W  # basic grid-world MDPs
import solver as S  # MDP solver (value-iteration)

# Define model parameters
states = ["Very low","Low","Medium","High","Very high"]
actions = ["Low","Medium","High"]
num_states = len(states)
num_actions = len(actions)
reward = {
    'Very low': reward_maxent[0][0],
    'Low': reward_maxent[0][1],
    'Medium': reward_maxent[0][2],
    'High': reward_maxent[0][3],
    'Very high': reward_maxent[0][4]
}

# Model parameterization
P = pp_transition # Probability transitions
gamma = 0.1  # Discount factor
epsilon = 0.01  # Convergence threshold

# Initialize the value function
V = {s: 0 for s in states}

# Define the required classes. For more details see Luz, M. (2019) at https://github.com/qzed/irl-maxent
def value_iteration(states, actions, reward, P, gamma, epsilon):
    num_states = len(states)
    num_actions = len(actions)
    delta = float('inf')
    while delta > epsilon:
        delta = 0
        for s_idx, s in enumerate(states):
            v = V[s]
            V[s] = reward[s] + gamma * max(
                sum(P[s_idx, s_prime_idx, a_idx] * V[states[s_prime_idx]] for s_prime_idx in range(num_states))
                for a_idx in range(num_actions)
            )
            delta = max(delta, abs(v - V[s]))
    return V

def extract_policy(states, actions, reward, P, V, gamma):
    num_states = len(states)
    num_actions = len(actions)
    policy = {}
    for s_idx, s in enumerate(states):
        policy[s] = actions[
            np.argmax([
                sum(P[s_idx, s_prime_idx, a_idx] * V[states[s_prime_idx]] for s_prime_idx in range(num_states))
                for a_idx in range(num_actions)
            ])
        ]
    return policy

# Run value iteration
V = value_iteration(states, actions, reward, P, gamma, epsilon)

# Extract the optimal policy
policy = extract_policy(states, actions, reward, P, V, gamma)


##### Print the results
## Print weights with their corresponding labels
labels = ['Day of the week', 'International day', 'COVID-19 deaths', 'COVID-19 cases',
          'Length of tweets', 'Weekly tweet count', 'URL', 'Hashtag', 'Mention']

print("\nFeature weights:")
for label, weight in zip(labels, reward_maxent[1]):
    print(f"{label}: {weight:.6f}")

print("\nOptimal Value Function:") # the expected cumulative reward starting from state s 
for s in states:
    print(f"V({s}) = {V[s]:.2f}")

print("\nOptimal Policy:")
for s in states:
    print(f"pi({s}) = {policy[s]}")
