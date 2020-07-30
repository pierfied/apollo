#ifndef APOLLO_MODEL_FACTORY_H
#define APOLLO_MODEL_FACTORY_H

#include <memory>
#include <vector>
#include "apollo/PolicyModel.h"
#include "apollo/TimingModel.h"

// Factory
class ModelFactory {
    public:
        static std::unique_ptr<PolicyModel> createStatic(int num_policies, int policy_choice );
        static std::unique_ptr<PolicyModel> createRandom(int num_policies);
        static std::unique_ptr<PolicyModel> createRoundRobin(int num_policies);

        static std::unique_ptr<PolicyModel> createDecisionTree(int num_policies,
                std::vector< std::vector<float> > &features,
                std::vector<int> &responses );

        static std::unique_ptr<TimingModel> createRegressionTree(
                std::vector< std::vector<float> > &features,
                std::vector<float> &responses );

        static std::unique_ptr<PolicyModel> createPolicyNet(int numPolicies, int numFeatures, double lr, double beta,
                double beta1, double beta2, double featureScaling, double threshold);
}; //end: ModelFactory


#endif
