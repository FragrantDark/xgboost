//
// Created by zhangqi on 2020-05-28.
//

#ifndef XGBOOST_LAB_API_H
#define XGBOOST_LAB_API_H

#include <vector>
#include <utility>
#include <unordered_map>
#include <string>
#include <list>
#include <xgboost/c_api.h>

namespace dce_lab {

class Sample {
public:
    float label;
    float weight;
    std::vector<std::pair<int, float> > features;

    Sample(float l, float w) : label(l), weight(w) {}

    void Clear() { features.clear(); }

    void AddFeature(int idx, float fvalue) { features.emplace_back(idx, fvalue); }
};

using sample_vec_t = std::vector<Sample>;
using param_dic_t = std::unordered_map<std::string, std::string>;
using pred_res_t = std::vector<float>;

using dataset_t = std::list<std::pair<float, std::list<std::pair<int, int> > > >;   // list<weight, list<fidx, fvalue>>

class XGB {
public:
    XGB(int n_col) : n_col_(n_col) {}
    ~XGB();

    /**
     *
     * @param train
     * @param test
     * @param param_dict    https://xgboost.readthedocs.io/en/latest/parameter.html
     * @param my_param  train_iteration
     * @return
     */
    int Train(const sample_vec_t& train, const sample_vec_t& test,
            const param_dic_t& param_dict, const dce_lab::param_dic_t& my_param);

    int Train(const dataset_t& pos_train, const dataset_t& neg_train,
            const dataset_t& pos_test, const dataset_t& neg_test,
            const param_dic_t& param_dict, const dce_lab::param_dic_t& my_param);

    int Predict(const sample_vec_t& test, pred_res_t& result) const;

    int Predict(const dataset_t& pos_test, const dataset_t& neg_test, pred_res_t& pos_res, pred_res_t& neg_res);

    int Save(const char* fname) const;

    int Load(const char* fname, const sample_vec_t& samples);

private:
    BoosterHandle booster;
    int n_col_;
};

}

#endif //XGBOOST_LAB_API_H
