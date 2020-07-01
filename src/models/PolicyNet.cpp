// Copyright (c) 2019, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory
//
// This file is part of Apollo.
// OCEC-17-092
// All rights reserved.
//
// Apollo is currently developed by Chad Wood, wood67@llnl.gov, with the help
// of many collaborators.
//
// Apollo was originally created by David Beckingsale, david@llnl.gov
//
// For details, see https://github.com/LLNL/apollo.
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.

#include <random>
#include <iostream>
#include "apollo/models/PolicyNet.h"

PolicyNet::PolicyNet(int numPolicies, int numFeatures, double lr = 1e-2, double beta = 0.5, double beta1 = 0.5,
                     double beta2 = 0.9, double featureScaling = 64 * std::log(2.)) :
        PolicyModel(numPolicies, "PolicyNet", true), numPolicies(numPolicies),
        net(numFeatures, (numFeatures + numPolicies) / 2, numPolicies, lr, beta1, beta2),
        beta(beta), featureScaling(featureScaling) {
    // Seed the random number generator using the current time.
    gen.seed(std::chrono::system_clock::now().time_since_epoch().count());
}

PolicyNet::~PolicyNet() {
}

void
PolicyNet::trainNet(std::vector<std::vector<float>> &states, std::vector<int> &actions, std::vector<double> &rewards) {
    int batchSize = states.size();
    if (batchSize < 1) return; // Don't train if there is no data to train on.
    int inputSize = states[0].size();

    // Create the arrays used for training.
    double *trainStates = new double[batchSize * inputSize];
    int *trainActions = new int[batchSize];
    double *trainRewards = new double[batchSize];

    // Calculate the average reward of the batch.
    double batchRewardAvg = 0;
    for (int i = 0; i < batchSize; ++i) {
        batchRewardAvg += rewards[i];
    }
    batchRewardAvg /= batchSize;

    // Update the moving average of the reward.
    rewardMovingAvg = beta * rewardMovingAvg + (1 - beta) * batchRewardAvg;

    // Fill the arrays used for training.
    for (int i = 0; i < batchSize; ++i) {
        for (int j = 0; j < inputSize; ++j) {
            trainStates[i * inputSize + j] = std::log(states[i][j]) / featureScaling; // Use log to normalize the feature scales.
        }
        trainActions[i] = actions[i];
        trainRewards[i] = rewards[i] - rewardMovingAvg; // Subtract the moving average baseline to reduce variance.
    }

    // Train the network.
    net.trainStep(trainStates, trainActions, trainRewards, batchSize);

    // Delete the arrays used for training.
    delete[] trainStates;
    delete[] trainActions;
    delete[] trainRewards;
}

int PolicyNet::getIndex(std::vector<float> &state) {
    std::vector<double> actionProbs;

    // Check if these features have already been evaluated since the previous network update.
    auto it = cache2.find(state[0]);
    if (it != cache2.end()) {
        // Use the previously evaluated action probabilities.
        actionProbs = it->second;
    } else {
        // Create the state array to be evaluated by the network.
        int inputSize = state.size();
        double *evalState = new double[inputSize];
        for (int i = 0; i < inputSize; ++i) {
            evalState[i] = std::log(state[i]) / featureScaling; // Use log to normalize the feature scales.
        }

        // Compute the action probabilities using the network and store in a vector.
        double *evalActionProbs = net.forward(evalState, 1);
        actionProbs = std::vector<double>(numPolicies);
        for (int i = 0; i < numPolicies; ++i) {
            actionProbs[i] = evalActionProbs[i];
        }

        // Delete the state array.
        delete[] evalState;

        // Add the action probabilities to the cache to be reused later.
        cache2.insert(std::make_pair(state[0], actionProbs));

//        std::cout << " probs: ";
//        for(auto &p: actionProbs){
//            std::cout << p << " ";
//        }
//        std::cout << std::endl;
    }

    // Sample a policy from the action probabilities.
    std::discrete_distribution<> d(actionProbs.begin(), actionProbs.end());
    int policyIndex = d(gen);

    return policyIndex;
}

//TODO: Implement network saving.
void PolicyNet::store(const std::string &filename) {

}
