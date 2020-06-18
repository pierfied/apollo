#ifndef APOLLO_MODELS_POLICYNET_H
#define APOLLO_MODELS_POLICYNET_H

#include <string>
#include <vector>
#include <random>
#include <unordered_map>
#include <map>

#include "apollo/PolicyModel.h"

class FCLayer {
public:
    FCLayer(int inputSize, int outputSize) : inputSize(inputSize), outputSize(outputSize) {
        std::default_random_engine generator;
        std::normal_distribution<double> distribution(0, 0.01);

        for (int i = 0; i < outputSize; ++i) {
            std::vector<double> row;
            for (int j = 0; j < inputSize; ++j) {
                row.push_back(distribution(generator));
            }

            weights.push_back(row);
            bias.push_back(distribution(generator));
        }
    };

    std::vector<std::vector<double>> forward(std::vector<std::vector<double>> &inputs) {
        int batchSize = inputs.size();

        std::vector<std::vector<double>> outputs;
        for (int i = 0; i < batchSize; ++i) {
            outputs.push_back(std::vector<double>(outputSize, 0));
        }

        for (int i = 0; i < batchSize; ++i) {
            for (int j = 0; j < outputSize; ++j) {
                for (int k = 0; k < inputSize; ++k) {
                    outputs[i][j] += weights[j][k] * inputs[i][k];
                }

                outputs[i][j] += bias[j];
            }
        }

        return outputs;
    }

    std::vector<std::vector<double>>
    backward(std::vector<std::vector<double>> &inputs, std::vector<std::vector<double>> &chainRuleGrad) {
        int batchSize = inputs.size();

        weight_grad = std::vector<std::vector<double>>();
        bias_grad = std::vector<double>(outputSize, 0);
        for (int i = 0; i < outputSize; ++i) {
            weight_grad.push_back(std::vector<double>(inputSize, 0));
        }

        std::vector<std::vector<double>> inputGrad;

        for (int i = 0; i < batchSize; ++i) {
            inputGrad.push_back(std::vector<double>(inputSize, 0));

            for (int j = 0; j < outputSize; ++j) {
                for (int k = 0; k < inputSize; ++k) {
                    weight_grad[j][k] += chainRuleGrad[i][j] * inputs[i][k];
                    inputGrad[i][k] += chainRuleGrad[i][j] * weights[j][k];
                }

                bias_grad[j] += chainRuleGrad[i][j];
            }
        }

        return inputGrad;
    }

    void step(double learnRate){
        for (int i = 0; i < outputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                weights[i][j] += learnRate * weight_grad[i][j];
            }

            bias[i] += learnRate * bias_grad[i];
        }
    }

private:
    int inputSize;
    int outputSize;
    std::vector<std::vector<double>> weights;
    std::vector<double> bias;
    std::vector<std::vector<double>> weight_grad;
    std::vector<double> bias_grad;
};

class Relu {
public:
    std::vector<std::vector<double>> forward(std::vector<std::vector<double>> &inputs) {
        int batchSize = inputs.size();
        int outputSize = inputs[0].size();

        std::vector<std::vector<double>> outputs;

        for (int i = 0; i < batchSize; ++i) {
            auto output = std::vector<double>(outputSize);
            for (int j = 0; j < outputSize; ++j) {
                output[j] = std::max(inputs[i][j], 0.);
            }

            outputs.push_back(output);
        }

        return outputs;
    }

    std::vector<std::vector<double>>
    backward(std::vector<std::vector<double>> &inputs, std::vector<std::vector<double>> &chainRuleGrad) {
        int batchSize = inputs.size();
        int inputSize = inputs[0].size();

        std::vector<std::vector<double>> inputGrad;

        for (int i = 0; i < batchSize; ++i) {
            inputGrad.push_back(std::vector<double>(inputSize, 0));

            for (int j = 0; j < inputSize; ++j) {
                inputGrad[i][j] = chainRuleGrad[i][j] * std::signbit(-inputs[i][j]);
            }
        }

        return inputGrad;
    }
};

class Softmax {
public:
    std::vector<std::vector<double>> forward(std::vector<std::vector<double>> &inputs) {
        int batchSize = inputs.size();
        int outputSize = inputs[0].size();

        std::vector<double> alpha(batchSize, 0);
        std::vector<std::vector<double>> outputs;

        for (int i = 0; i < batchSize; ++i) {
            outputs.push_back(std::vector<double>(outputSize, 0));
        }

        for (int i = 0; i < batchSize; ++i) {
            alpha[i] = inputs[i][0];
            for (int j = 0; j < outputSize; ++j) {
                alpha[i] = std::max(alpha[i], inputs[i][j]);
            }
        }

        for (int i = 0; i < batchSize; ++i) {
            double sum = 0;
            for (int j = 0; j < outputSize; ++j) {
                outputs[i][j] = std::exp(inputs[i][j] - alpha[i]);
                sum += outputs[i][j];
            }
            for (int j = 0; j < outputSize; ++j) {
                outputs[i][j] /= sum;
            }
        }

        return outputs;
    }

    std::vector<std::vector<double>> lossGrad(std::vector<std::vector<double>> &input, std::vector<int> &action,
                                              std::vector<std::vector<double>> &actionProbs,
                                              std::vector<double> &reward) {
        int batchSize = input.size();
        int inputSize = input[0].size();

        std::vector<std::vector<double>> inputGrad;
        for (int i = 0; i < batchSize; ++i) {
            inputGrad.push_back(std::vector<double>(inputSize, 0));
            for (int j = 0; j < inputSize; ++j) {
                inputGrad[i][j] = -reward[i] * actionProbs[i][j] / batchSize;
            }
            inputGrad[i][action[i]] += reward[i] / batchSize;
        }

        return inputGrad;
    }
};

class Net {
public:
    Net(int inputSize, int hiddenSize, int outputSize, double learnRate=1e-1)
            : inputSize(inputSize), hiddenSize(hiddenSize), outputSize(outputSize), learnRate(learnRate),
              layer1(FCLayer(inputSize, hiddenSize)), layer2(FCLayer(hiddenSize, outputSize)),
              layer3(FCLayer(hiddenSize, outputSize)) {}

    std::vector<std::vector<double>> forward(std::vector<std::vector<double>> &input){
        auto out = layer1.forward(input);
        out = relu.forward(out);
        out = layer2.forward(out);
        out = relu.forward(out);
        out = layer3.forward(out);
        out = softmax.forward(out);

        return out;
    }

    void trainStep(std::vector<std::vector<double>> &state, std::vector<int> &action, std::vector<double> &reward){
        auto out1 = layer1.forward(state);
        auto aout1 = relu.forward(out1);
        auto out2 = layer2.forward(aout1);
        auto aout2 = relu.forward(out2);
        auto out3 = layer3.forward(aout2);
        auto aout3 = softmax.forward(out3);

        auto aout3_grad = softmax.lossGrad(out3, action, aout3, reward);
        auto out3_grad = layer3.backward(aout2, aout3_grad);
        auto aout2_grad = relu.backward(out2, out3_grad);
//        auto aout2_grad = softmax.lossGrad(out2, action, aout2, reward);
        auto out2_grad = layer2.backward(aout1, aout2_grad);
        auto aout1_grad = relu.backward(out1, out2_grad);
        layer1.backward(state, aout1_grad);

        layer1.step(learnRate);
        layer2.step(learnRate);
        layer3.step(learnRate);
    }

private:
    int inputSize;
    int hiddenSize;
    int outputSize;
    double learnRate;
    FCLayer layer1;
    FCLayer layer2;
    FCLayer layer3;
    Relu relu;
    Softmax softmax;
};

class PolicyNet : public PolicyModel {
public:
    PolicyNet(int num_policies, int num_features);

    ~PolicyNet();

    int getIndex(std::vector<float> &state);

    void trainNet(std::vector<std::vector<float>> &states, std::vector<int> &actions, std::vector<double> &rewards);

    void store(const std::string &filename);

    std::map<std::vector<float>, std::vector<double>> cache;
    std::unordered_map<float, std::vector<double>> cache2;
private:
    int numPolicies;
    Net net;
    std::default_random_engine gen;
}; //end: PolicyNet (class)


#endif
