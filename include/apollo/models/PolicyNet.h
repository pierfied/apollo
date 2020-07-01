#ifndef APOLLO_MODELS_POLICYNET_H
#define APOLLO_MODELS_POLICYNET_H

#include <string>
#include <vector>
#include <random>
#include <unordered_map>
#include <map>
#include <chrono>

#include "apollo/PolicyModel.h"

class FCLayer {
public:
    FCLayer(int inputSize, int outputSize) : inputSize(inputSize), outputSize(outputSize) {
        // Create the random number generator and the distribution for He initialization.
        std::mt19937_64 generator(std::chrono::system_clock::now().time_since_epoch().count());
        double stddev = std::sqrt(2. / inputSize);
        std::normal_distribution<double> distribution(0., stddev);

        // Allocate and zero initialize the arrays needed for the parameters.
        weights = new double[outputSize * inputSize]();
        weights_m = new double[outputSize * inputSize]();
        weights_v = new double[outputSize * inputSize]();
        weights_grad = new double[outputSize * inputSize]();

        bias = new double[outputSize]();
        bias_m = new double[outputSize]();
        bias_v = new double[outputSize]();
        bias_grad = new double[outputSize]();

        // Randomly initializer the layer weights.
        for (int i = 0; i < outputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                weights[i * inputSize + j] = distribution(generator);
            }

//            bias[i] = distribution(generator);
        }
    };

    ~FCLayer() {
        // Delete all allocated arrays.
        delete[] weights;
        delete[] weights_m;
        delete[] weights_v;
        delete[] weights_grad;

        delete[] bias;
        delete[] bias_m;
        delete[] bias_v;
        delete[] bias_grad;

        if (outputs != NULL) {
            delete[] outputs;
            delete[] inputGrad;
        }
    }

    double *forward(double *inputs, int batchSize) {
        // Check if the current batch size is larger than the maximum so far.
        if (batchSize > maxBatchSize) {
            // Delete the old arrays if they exist.
            if (outputs != NULL) {
                delete[] outputs;
                delete[] inputGrad;
            }

            // Allocate new arrays for the batch size.
            outputs = new double[batchSize * outputSize]();
            inputGrad = new double[batchSize * inputSize];

            maxBatchSize = batchSize;
        } else {
            // Zero the output array.
            for (int i = 0; i < batchSize * outputSize; ++i) {
                outputs[i] = 0;
            }
        }

        // Compute the forward pass of this layer: y = M x + b
        for (int i = 0; i < batchSize; ++i) {
            for (int j = 0; j < outputSize; ++j) {
                for (int k = 0; k < inputSize; ++k) {
                    outputs[i * outputSize + j] += weights[j * inputSize + k] * inputs[i * inputSize + k];
                }

                outputs[i * outputSize + j] += bias[j];
            }
        }

        return outputs;
    }

    double *backward(double *inputs, double *chainRuleGrad, int batchSize) {
        // Zero all gradient arrays.
        for (int i = 0; i < batchSize * inputSize; ++i) {
            inputGrad[i] = 0;
        }

        for (int i = 0; i < outputSize * inputSize; ++i) {
            weights_grad[i] = 0;
        }

        for (int i = 0; i < outputSize; ++i) {
            bias_grad[i] = 0;
        }

        // Compute the gradients using the chain rule with the gradients from the previous layer.
        for (int i = 0; i < batchSize; ++i) {
            for (int j = 0; j < outputSize; ++j) {
                for (int k = 0; k < inputSize; ++k) {
                    weights_grad[j * inputSize + k] += chainRuleGrad[i * outputSize + j] * inputs[i * inputSize + k];
                    inputGrad[i * inputSize + k] += chainRuleGrad[i * outputSize + j] * weights[j * inputSize + k];
                }

                bias_grad[j] += chainRuleGrad[i * outputSize + j];
            }
        }

        return inputGrad;
    }

    void step(double learnRate, double beta1 = 0.5, double beta2 = 0.9, double epsilon = 1e-8) {
        stepNum++;

        // Update each parameter in the layer using Adam optimzation.
        for (int i = 0; i < outputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                int index = i * inputSize + j;

                // Update the moving averages of the first and second moments of the gradient.
                weights_m[index] = beta1 * weights_m[index] + (1 - beta1) * weights_grad[index];
                weights_v[index] = beta2 * weights_v[index] + (1 - beta2) * std::pow(weights_grad[index], 2);

                // Compute the unbiased moving averages.
                double mHat = weights_m[index] / (1 - std::pow(beta1, stepNum));
                double vHat = weights_v[index] / (1 - std::pow(beta2, stepNum));

                // Update the weights.
                weights[index] += learnRate * mHat / (std::sqrt(vHat) + epsilon);
            }

            // Update the moving averages of the first and second moments of the gradient.
            bias_m[i] = beta1 * bias_m[i] + (1 - beta1) * bias_grad[i];
            bias_v[i] = beta2 * bias_v[i] + (1 - beta2) * std::pow(bias_grad[i], 2);

            // Compute the unbiased moving averages.
            double mHat = bias_m[i] / (1 - std::pow(beta1, stepNum));
            double vHat = bias_v[i] / (1 - std::pow(beta2, stepNum));

            // Update the weights.
            bias[i] += learnRate * mHat / (std::sqrt(vHat) + epsilon);
        }
    }

private:
    int inputSize;
    int outputSize;
    long stepNum = 0;

    double *weights;
    double *weights_m;
    double *weights_v;
    double *weights_grad;
    double *bias;
    double *bias_m;
    double *bias_v;
    double *bias_grad;

    double *outputs = NULL;
    double *inputGrad;
    int maxBatchSize = 0;
};

class Relu {
public:
    ~Relu() {
        // Delete allocated arrays.
        if (outputs != NULL) {
            delete[] outputs;
            delete[] inputGrad;
        }
    }

    double *forward(double *inputs, int batchSize, int outputSize) {
        // Check if the arrays need to be reallocated.
        if (batchSize * outputSize > maxArraySize) {
            maxArraySize = batchSize * outputSize;

            // Delete old arrays if they exist.
            if (outputs != NULL) {
                delete[] outputs;
                delete[] inputGrad;
            }

            outputs = new double[maxArraySize];
            inputGrad = new double[maxArraySize];
        }

        // Compute ReLU activation.
        for (int i = 0; i < batchSize * outputSize; ++i) {
            outputs[i] = std::max(inputs[i], 0.);
        }

        return outputs;
    }

    double *backward(double *inputs, double *chainRuleGrad, int batchSize, int inputSize) {
        // Compute the gradients of the ReLU layer using the chain rule with the gradients from the previous layer.
        for (int i = 0; i < batchSize * inputSize; ++i) {
            inputGrad[i] = chainRuleGrad[i] * std::signbit(-inputs[i]);
        }

        return inputGrad;
    }

private:
    double *outputs = NULL;
    double *inputGrad;
    int maxArraySize = 0;
};

class Softmax {
public:
    ~Softmax() {
        // Delete allocated arrays.
        if (outputs != NULL) {
            delete[] outputs;
            delete[] inputGrad;
        }
    }

    double *forward(double *inputs, int batchSize, int outputSize) {
        // Check if the arrays need to be reallocated.
        if (batchSize * outputSize > maxArraySize) {
            maxArraySize = batchSize * outputSize;

            // Delete old arrays if they exist.
            if (outputs != NULL) {
                delete[] outputs;
                delete[] inputGrad;
            }

            outputs = new double[maxArraySize];
            inputGrad = new double[maxArraySize];
        }

        // Compute the maximum of each previous layer for each sample. This is used to improve numerical stability.
        double alpha[batchSize];
        for (int i = 0; i < batchSize; ++i) {
            alpha[i] = inputs[i * outputSize];
            for (int j = 0; j < outputSize; ++j) {
                alpha[i] = std::max(alpha[i], inputs[i * outputSize + j]);
            }
        }

        // Calculate the probabilities for each output using softmax.
        for (int i = 0; i < batchSize; ++i) {
            double sum = 0;
            for (int j = 0; j < outputSize; ++j) {
                int index = i * outputSize + j;
                outputs[index] = std::exp(inputs[index] - alpha[i]); // Note: subtract alpha for numerical stability.
                sum += outputs[index];
            }
            for (int j = 0; j < outputSize; ++j) {
                outputs[i * outputSize + j] /= sum;
            }
        }

        return outputs;
    }

    double *lossGrad(int *action, double *actionProbs, double *reward, int batchSize, int inputSize) {
        // Compute the gradient of the softmax layer for the given state, action, and reward.
        for (int i = 0; i < batchSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                int index = i * inputSize + j;
                inputGrad[index] = -reward[i] * actionProbs[index] / batchSize;
            }
            inputGrad[i * inputSize + action[i]] += reward[i] / batchSize;
        }

        return inputGrad;
    }

private:
    double *outputs = NULL;
    double *inputGrad;
    int maxArraySize = 0;
};

class Net {
public:
    Net(int inputSize, int hiddenSize, int outputSize, double learnRate = 1e-2, double beta1 = 0.5, double beta2 = 0.9)
            : inputSize(inputSize), hiddenSize(hiddenSize), outputSize(outputSize), learnRate(learnRate),
              beta1(beta1), beta2(beta2), layer1(FCLayer(inputSize, hiddenSize)),
              layer2(FCLayer(hiddenSize, hiddenSize)), layer3(FCLayer(hiddenSize, outputSize)) {}

    double *forward(double *state, int batchSize) {
        // Compute the forward pass through each layer of the network.
        auto out1 = layer1.forward(state, batchSize);
        auto aout1 = relu1.forward(out1, batchSize, hiddenSize);
        auto out2 = layer2.forward(aout1, batchSize);
        auto aout2 = relu2.forward(out2, batchSize, hiddenSize);
        auto out3 = layer3.forward(aout2, batchSize);
        auto aout3 = softmax.forward(out3, batchSize, outputSize);

        return aout3;
    }

    void trainStep(double *state, int *action, double *reward, int batchSize) {
        // Compute the forward pass through each layer of the network.
        auto out1 = layer1.forward(state, batchSize);
        auto aout1 = relu1.forward(out1, batchSize, hiddenSize);
        auto out2 = layer2.forward(aout1, batchSize);
        auto aout2 = relu2.forward(out2, batchSize, hiddenSize);
        auto out3 = layer3.forward(aout2, batchSize);
        auto aout3 = softmax.forward(out3, batchSize, outputSize);

        // Compute the backward pass through each layer and compute the gradients.
        auto aout3_grad = softmax.lossGrad(action, aout3, reward, batchSize, outputSize);
        auto out3_grad = layer3.backward(aout2, aout3_grad, batchSize);
        auto aout2_grad = relu2.backward(out2, out3_grad, batchSize, hiddenSize);
        auto out2_grad = layer2.backward(aout1, aout2_grad, batchSize);
        auto aout1_grad = relu1.backward(out1, out2_grad, batchSize, hiddenSize);
        layer1.backward(state, aout1_grad, batchSize);

        // Update the parameters of each layer.
        layer1.step(learnRate, beta1, beta2);
        layer2.step(learnRate, beta1, beta2);
        layer3.step(learnRate, beta1, beta2);
    }

private:
    int inputSize;
    int hiddenSize;
    int outputSize;
    double learnRate;
    double beta1;
    double beta2;
    FCLayer layer1;
    FCLayer layer2;
    FCLayer layer3;
    Relu relu1;
    Relu relu2;
    Softmax softmax;
};

class PolicyNet : public PolicyModel {
public:
    PolicyNet(int num_policies, int num_features, double lr, double beta, double beta1, double beta2,
            double featureScaling);

    ~PolicyNet();

    int getIndex(std::vector<float> &state);

    void trainNet(std::vector<std::vector<float>> &states, std::vector<int> &actions, std::vector<double> &rewards);

    void store(const std::string &filename);

    std::map<std::vector<float>, std::vector<double>> cache;
    std::unordered_map<float, std::vector<double>> cache2;
private:
    int numPolicies;
    Net net;
    std::mt19937_64 gen;
    double rewardMovingAvg;
    double beta;
    double featureScaling;
}; //end: PolicyNet (class)


#endif
