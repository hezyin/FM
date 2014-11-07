#pragma GCC diagnostic ignored "-Wunused-result"

#ifndef _COMMON_H_
#define _COMMON_H_

#define flag { printf("\nLINE: %d\n", __LINE__); fflush(stdout); }

#include <vector>
#include <cmath>

struct Problem
{
    Problem(uint32_t const nr_instance, uint32_t const nr_field) 
        : nr_feature(0), nr_instance(nr_instance), nr_field(nr_field), 
          v(2.0f/static_cast<float>(nr_field)), 
          J(static_cast<uint64_t>(nr_instance)*nr_field), 
          Y(nr_instance) {}
    uint32_t nr_feature, nr_instance, nr_field;
    float v;
    std::vector<uint32_t> J; // J stores training dataset of size nr_instance * nr_field
    std::vector<float> Y; // Y stores each instance's label
};

Problem read_problem(std::string const path);

uint32_t const kW_NODE_SIZE = 2;

struct Model
{
    Model(uint32_t const nr_feature, uint32_t const nr_factor, uint32_t const nr_field) 
        : W(static_cast<uint64_t>(nr_feature)*nr_field*nr_factor*kW_NODE_SIZE, 0), 
          nr_feature(nr_feature), nr_factor(nr_factor), nr_field(nr_field) {}
    std::vector<float> W;
    uint32_t nr_feature, nr_factor, nr_field;
};

FILE *open_c_file(std::string const &path, std::string const &mode);

std::vector<std::string> 
argv_to_args(int const argc, char const * const * const argv);

inline float wTx(Problem const &prob, Model &model, uint32_t const i, 
    float const kappa=0, float const eta=0, float const lambda=0, 
    bool const do_update=false)
{
    uint32_t const nr_factor = model.nr_factor;
    uint32_t const nr_field = model.nr_field;
    uint32_t const nr_feature = model.nr_feature;
    uint64_t const align0 = nr_factor*kW_NODE_SIZE; // +align0 := next nr_field
    uint64_t const align1 = nr_field*align0; // +align1 := next nr_feature

    uint32_t const * const J = &prob.J[i*nr_field]; 
    float * const W = model.W.data();

    float const v = prob.v;
    float const kappav = kappa * v;
    float predict = 0.0;

    for(uint32_t f1 = 0; f1 < nr_field; ++f1)
    {
        uint32_t const j1 = J[f1]; 
        if(j1 >= nr_feature)
            continue;

        for(uint32_t f2 = f1+1; f2 < nr_field; ++f2)
        {
            uint32_t const j2 = J[f2];
            if(j2 >= nr_feature)
                continue;

            float * const w1 = W + j1*align1 + f2*align0; // W_{j1, f2}
            float * const w2 = W + j2*align1 + f1*align0; // W_{j2, f1}
          
            if(do_update)
            {
                float * const sg1 = w1 + nr_factor; // point to sum of history gradients
                float * const sg2 = w2 + nr_factor;
                for(uint32_t d = 0; d < nr_factor; d++)
                {
                    float const w1d = *(w1 + d); //dth element of W_{j1, f2}
                    float const w2d = *(w2 + d);
                    float const sg1d = *(sg1 + d);// history gradients for dth element
                    float const sg2d = *(sg2 + d);
                    float const g1 = kappav * w2d + lambda * w1d; 
                    float const g2 = kappav * w1d + lambda * w2d;
                    *(sg1 + d) = sg1d + g1 * g1; // update history gradients
                    *(sg2 + d) = sg2d + g2 * g2;
                    *(w1 + d) = static_cast<float> (w1d - eta * g1 / sqrt(*(sg1 + d))); //update parameters
                    *(w2 + d) = static_cast<float> (w2d - eta * g2 / sqrt(*(sg2 + d)));
                }
            }
            else
            {
                for(uint32_t d = 0; d < nr_factor; d++)
                {
                    predict += *(w1 + d) * *(w2 + d) * v;
                }
            }
        }
    }

    if(do_update)
        return 0;
    return predict;
}

float predict(Problem const &prob, Model &model, 
    std::string const &output_path = std::string(""));
#endif // _COMMON_H_
