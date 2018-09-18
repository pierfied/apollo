
#ifndef APOLLO_REGION_H
#define APOLLO_REGION_H

#include "apollo/Apollo.h"
#include "apollo/ModelWrapper.h"

#include "caliper/cali.h"
#include "caliper/Annotation.h"

class Apollo::Region {
    public:
        Region( Apollo       *apollo,
                const char   *regionName,
                int           numAvailablePolicies);
        ~Region();

        // Forward declarations:
        class Feature;

        //
        void begin(void); //Default begin(): f.hint != Feature::Hint.DEPENDANT
        //
        void iterationStart(int i);
        void iterationStop(void);
        //
        void end(void);   //Default end(): f.hint == Feature::Hint.DEPENDANT
        //
        Apollo::ModelWrapper *getModel(void);
        //
        void caliSetInt(const char *name, int value);
        void caliSetString(const char *name, const char *value);
        //
        char          *name;
 
    private:
        void handleCommonBeginTasks(void);
        void handleCommonEndTasks(void);
        //
        Apollo        *apollo;
        //
        //
        // Deprecated (somewhat):
        uint64_t       id;
        uint64_t       parent_id;
        bool           ynInsideMarkedRegion;
        char           CURRENT_BINDING_GUID[256];
        //
        Apollo::ModelWrapper  *model;
        cali::Loop            *cali_obj;
        cali::Loop::Iteration *cali_iter_obj;
}; //end: Apollo::Region


#endif