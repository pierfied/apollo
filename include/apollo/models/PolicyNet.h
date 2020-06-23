#ifndef APOLLO_MODELS_POLICYNET_H
#define APOLLO_MODELS_POLICYNET_H

#include <string>
#include <vector>
#include <random>
#include <unordered_map>
#include <map>
#include <chrono>

//#include "apollo/UmpireTools.h"
#include "apollo/PolicyModel.h"
#include "umpire/Umpire.hpp"

template<typename T>
T *poolAlloc(int size);
void poolFree(void *data);

class UmpirePool {
public:
    umpire::Allocator allocator;
    umpire::Allocator pooledAllocator;

    UmpirePool() {
        auto &rm = umpire::ResourceManager::getInstance();
        allocator = rm.getAllocator("HOST");
        pooledAllocator = rm.makeAllocator<umpire::strategy::DynamicPool>("HOST_pool", allocator);
    }
};

class StaticUmpirePool {
public:
    static UmpirePool pool;
};

class FCLayer {
public:
    FCLayer(int inputSize, int outputSize) : inputSize(inputSize), outputSize(outputSize) {
        std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
        double bound = std::sqrt(6. / (inputSize + outputSize)) * 0.1;
        std::uniform_real_distribution<double> distribution(-bound, bound);

        weights = poolAlloc<double>(inputSize * outputSize);
        weights_m = poolAlloc<double>(inputSize * outputSize);
        weights_v = poolAlloc<double>(inputSize * outputSize);
        weights_grad = poolAlloc<double>(inputSize * outputSize);

        bias = poolAlloc<double>(outputSize);
        bias_m = poolAlloc<double>(outputSize);
        bias_v = poolAlloc<double>(outputSize);
        bias_grad = poolAlloc<double>(outputSize);

        for (int i = 0; i < outputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                int index = i * inputSize + j;

                weights[index] = distribution(generator);
                weights_m[index] = 0;
                weights_v[index] = 0;
            }

            bias[i] = distribution(generator);
            bias_m[i] = 0;
            bias_v[i] = 0;
        }
    };

    double *forward(double *inputs, int batchSize) {
        double *outputs = poolAlloc<double>(batchSize * outputSize);

        for (int i = 0; i < batchSize * outputSize; ++i) {
            outputs[i] = 0;
        }

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
        for (int i = 0; i < outputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                weights_grad[i * inputSize + j];
            }
            bias_grad[i] = 0;
        }

        double *inputGrad = poolAlloc<double>(batchSize * inputSize);
        for (int i = 0; i < batchSize * inputSize; ++i) {
            inputGrad[i] = 0;
        }

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

        for (int i = 0; i < outputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                weights_m[i * inputSize + j] =
                        beta1 * weights_m[i * inputSize + j] + (1 - beta1) * weights_grad[i * inputSize + j];
                weights_v[i * inputSize + j] = beta2 * weights_v[i * inputSize + j] +
                                               (1 - beta2) * std::pow(weights_grad[i * inputSize + j], 2);

                double mHat = weights_m[i * inputSize + j] / (1 - std::pow(beta1, stepNum));
                double vHat = weights_v[i * inputSize + j] / (1 - std::pow(beta2, stepNum));

                weights[i * inputSize + j] += learnRate * mHat / (std::sqrt(vHat) + epsilon);
            }

            bias_m[i] = beta1 * bias_m[i] + (1 - beta1) * bias_grad[i];
            bias_v[i] = beta2 * bias_v[i] + (1 - beta2) * std::pow(bias_grad[i], 2);

            double mHat = bias_m[i] / (1 - std::pow(beta1, stepNum));
            double vHat = bias_v[i] / (1 - std::pow(beta2, stepNum));

            bias[i] += learnRate * mHat / (std::sqrt(vHat) + epsilon);
        }
    }

private:
    int inputSize;
    int outputSize;
    long stepNum = 0;
//    std::vector<std::vector<double>> weights;
//    std::vector<std::vector<double>> weights_m;
//    std::vector<std::vector<double>> weights_v;
//    std::vector<double> bias;
//    std::vector<double> bias_m;
//    std::vector<double> bias_v;
//    std::vector<std::vector<double>> weight_grad;
//    std::vector<double> bias_grad;

    double *weights, *weights_m, *weights_v, *weights_grad;
    double *bias, *bias_m, *bias_v, *bias_grad;
};

class Relu {
public:
    double *forward(double *inputs, int batchSize, int outputSize) {
        double *outputs = poolAlloc<double>(batchSize * outputSize);

        for (int i = 0; i < batchSize * outputSize; ++i) {
            outputs[i] = std::max(inputs[i], 0.);
        }

        return outputs;
    }

    double *backward(double *inputs, double *chainRuleGrad, int batchSize, int inputSize) {
        double *inputGrad = poolAlloc<double>(batchSize * inputSize);

        for (int i = 0; i < batchSize * inputSize; ++i) {
            inputGrad[i] = chainRuleGrad[i] * std::signbit(-inputs[i]);
        }

        return inputGrad;
    }
};

class Softmax {
public:
    double *forward(double *inputs, int batchSize, int outputSize) {
        double *alpha = poolAlloc<double>(batchSize);
        double *outputs = poolAlloc<double>(batchSize * outputSize);

        for (int i = 0; i < batchSize; ++i) {
            alpha[i] = inputs[i * outputSize];
            for (int j = 0; j < outputSize; ++j) {
                alpha[i] = std::max(alpha[i], inputs[i * outputSize + j]);
            }
        }

        for (int i = 0; i < batchSize; ++i) {
            double sum = 0;
            for (int j = 0; j < outputSize; ++j) {
                outputs[i * outputSize + j] = std::exp(inputs[i * outputSize + j] - alpha[i]);
                sum += outputs[i * outputSize + j];
            }
            for (int j = 0; j < outputSize; ++j) {
                outputs[i * outputSize + j] /= sum;
            }
        }

        poolFree(alpha);

        return outputs;
    }

    double *lossGrad(double *input, int *action, double *actionProbs, double *reward, int batchSize, int inputSize) {
        double *inputGrad = poolAlloc<double>(batchSize * inputSize);

        for (int i = 0; i < batchSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                inputGrad[i * inputSize + j] = -reward[i] * actionProbs[i * inputSize + j] / batchSize;
            }
            inputGrad[i * inputSize + action[i]] += reward[i] / batchSize;
        }

        return inputGrad;
    }
};

class Net {
public:
    Net(int inputSize, int hiddenSize, int outputSize, double learnRate = 1e-1)
            : inputSize(inputSize), hiddenSize(hiddenSize), outputSize(outputSize), learnRate(learnRate),
              layer1(FCLayer(inputSize, hiddenSize)), layer2(FCLayer(hiddenSize, hiddenSize)),
              layer3(FCLayer(hiddenSize, outputSize)) {}

    double *forward(double *state, int batchSize) {
        auto out1 = layer1.forward(state, batchSize);
        auto aout1 = relu.forward(out1, batchSize, hiddenSize);
        auto out2 = layer2.forward(aout1, batchSize);
        auto aout2 = relu.forward(out2, batchSize, hiddenSize);
        auto out3 = layer3.forward(aout2, batchSize);
        auto aout3 = softmax.forward(out3, batchSize, outputSize);

        poolFree(out1);
        poolFree(aout1);
        poolFree(out2);
        poolFree(aout2);
        poolFree(out3);

        return aout3;
    }

    void trainStep(double *state, int *action, double *reward, int batchSize) {
        auto out1 = layer1.forward(state, batchSize);
        auto aout1 = relu.forward(out1, batchSize, hiddenSize);
        auto out2 = layer2.forward(aout1, batchSize);
        auto aout2 = relu.forward(out2, batchSize, hiddenSize);
        auto out3 = layer3.forward(aout2, batchSize);
        auto aout3 = softmax.forward(out3, batchSize, outputSize);

        auto aout3_grad = softmax.lossGrad(out3, action, aout3, reward, batchSize, outputSize);
        auto out3_grad = layer3.backward(aout2, aout3_grad, batchSize);
        auto aout2_grad = relu.backward(out2, out3_grad, batchSize, hiddenSize);
//        auto aout2_grad = softmax.lossGrad(out2, action, aout2, reward);
        auto out2_grad = layer2.backward(aout1, aout2_grad, batchSize);
        auto aout1_grad = relu.backward(out1, out2_grad, batchSize, hiddenSize);
        layer1.backward(state, aout1_grad, batchSize);

        poolFree(out1);
        poolFree(aout1);
        poolFree(out2);
        poolFree(aout2);
        poolFree(out3);
        poolFree(aout3);

        poolFree(aout3_grad);
        poolFree(out3_grad);
        poolFree(aout2_grad);
        poolFree(out2_grad);
        poolFree(aout1_grad);

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
