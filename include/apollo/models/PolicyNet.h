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
        std::mt19937_64 generator(std::chrono::system_clock::now().time_since_epoch().count());
        double stddev = std::sqrt(2 / inputSize);
        std::normal_distribution<double> distribution(0., stddev);

        weights = new double[outputSize * inputSize]();
        weights_m = new double[outputSize * inputSize]();
        weights_v = new double[outputSize * inputSize]();
        weights_grad = new double[outputSize * inputSize]();

        bias = new double[outputSize]();
        bias_m = new double[outputSize]();
        bias_v = new double[outputSize]();
        bias_grad = new double[outputSize]();

        for (int i = 0; i < outputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                weights[i * inputSize + j] = distribution(generator);
            }

//            bias[i] = distribution(generator);
        }
    };

    ~FCLayer(){
        delete weights;
        delete weights_m;
        delete weights_v;
        delete weights_grad;

        delete bias;
        delete bias_m;
        delete bias_v;
        delete bias_grad;

        if (outputs != NULL) {
            delete outputs;
            delete inputGrad;
        }
    }

    double *forward(double *inputs, int batchSize) {
        if (batchSize > maxBatchSize){
            if (outputs != NULL) {
                delete outputs;
                delete inputGrad;
            }

            outputs = new double[batchSize * outputSize]();
            inputGrad = new double[batchSize * inputSize];

            maxBatchSize = batchSize;
        }else{
            for (int i = 0; i < batchSize * outputSize; ++i) {
                outputs[i] = 0;
            }
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
        for (int i = 0; i < batchSize * inputSize; ++i) {
            inputGrad[i] = 0;
        }

        for (int i = 0; i < outputSize * inputSize; ++i) {
            weights_grad[i] = 0;
        }

        for (int i = 0; i < outputSize; ++i) {
            bias_grad[i] = 0;
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
                int index = i * inputSize + j;

                weights_m[index] = beta1 * weights_m[index] + (1 - beta1) * weights_grad[index];
                weights_v[index] = beta2 * weights_v[index] + (1 - beta2) * std::pow(weights_grad[index], 2);

                double mHat = weights_m[index] / (1 - std::pow(beta1, stepNum));
                double vHat = weights_v[index] / (1 - std::pow(beta2, stepNum));

                weights[index] += learnRate * mHat / (std::sqrt(vHat) + epsilon);
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
    ~Relu(){
        if (outputs != NULL) {
            delete outputs;
            delete inputGrad;
        }
    }

    double *forward(double *inputs, int batchSize, int outputSize) {
        if (batchSize * outputSize > maxArraySize){
            maxArraySize = batchSize * outputSize;

            if (outputs != NULL) {
                delete outputs;
                delete inputGrad;
            }

            outputs = new double[maxArraySize];
            inputGrad = new double[maxArraySize];
        }

        for (int i = 0; i < batchSize * outputSize; ++i) {
            outputs[i] = std::max(inputs[i], 0.);
        }

        return outputs;
    }

    double *backward(double *inputs, double *chainRuleGrad, int batchSize, int inputSize) {
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
    ~Softmax(){
        if (outputs != NULL) {
            delete outputs;
            delete inputGrad;
        }
    }

    double *forward(double *inputs, int batchSize, int outputSize) {
        if (batchSize * outputSize > maxArraySize){
            maxArraySize = batchSize * outputSize;

            if (outputs != NULL) {
                delete outputs;
                delete inputGrad;
            }

            outputs = new double[maxArraySize];
            inputGrad = new double[maxArraySize];
        }

        double alpha[batchSize];

        for (int i = 0; i < batchSize; ++i) {
            alpha[i] = inputs[i * outputSize];
            for (int j = 0; j < outputSize; ++j) {
                alpha[i] = std::max(alpha[i], inputs[i * outputSize + j]);
            }
        }

        for (int i = 0; i < batchSize; ++i) {
            double sum = 0;
            for (int j = 0; j < outputSize; ++j) {
                int index = i * outputSize + j;
                outputs[index] = std::exp(inputs[index] - alpha[i]);
                sum += outputs[index];
            }
            for (int j = 0; j < outputSize; ++j) {
                outputs[i * outputSize + j] /= sum;
            }
        }

        return outputs;
    }

    double *lossGrad(int *action, double *actionProbs, double *reward, int batchSize, int inputSize) {
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
    Net(int inputSize, int hiddenSize, int outputSize, double learnRate = 1e-1)
            : inputSize(inputSize), hiddenSize(hiddenSize), outputSize(outputSize), learnRate(learnRate),
              layer1(FCLayer(inputSize, hiddenSize)), layer2(FCLayer(hiddenSize, hiddenSize)),
              layer3(FCLayer(hiddenSize, outputSize)) {}

    double *forward(double *state, int batchSize) {
        auto out1 = layer1.forward(state, batchSize);
        auto aout1 = relu1.forward(out1, batchSize, hiddenSize);
        auto out2 = layer2.forward(aout1, batchSize);
        auto aout2 = relu2.forward(out2, batchSize, hiddenSize);
        auto out3 = layer3.forward(aout2, batchSize);
        auto aout3 = softmax.forward(out3, batchSize, outputSize);

        return aout3;
    }

    void trainStep(double *state, int *action, double *reward, int batchSize) {
        auto out1 = layer1.forward(state, batchSize);
        auto aout1 = relu1.forward(out1, batchSize, hiddenSize);
        auto out2 = layer2.forward(aout1, batchSize);
        auto aout2 = relu2.forward(out2, batchSize, hiddenSize);
        auto out3 = layer3.forward(aout2, batchSize);
        auto aout3 = softmax.forward(out3, batchSize, outputSize);

        auto aout3_grad = softmax.lossGrad(action, aout3, reward, batchSize, outputSize);
        auto out3_grad = layer3.backward(aout2, aout3_grad, batchSize);
        auto aout2_grad = relu2.backward(out2, out3_grad, batchSize, hiddenSize);
        auto out2_grad = layer2.backward(aout1, aout2_grad, batchSize);
        auto aout1_grad = relu1.backward(out1, out2_grad, batchSize, hiddenSize);
        layer1.backward(state, aout1_grad, batchSize);

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
    Relu relu1;
    Relu relu2;
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
    std::mt19937_64 gen;
}; //end: PolicyNet (class)


#endif
