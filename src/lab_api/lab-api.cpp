//
// Created by zhangqi on 2020-05-28.
//
#include <lab-api.h>
#include "../data/simple_dmatrix.h"

namespace dce_lab {

int SampleVec2SimpleDMatrix(const sample_vec_t& svec, xgboost::data::SimpleDMatrix& sdmtx) {
    // TODO
}

int XGB::Train(const dce_lab::sample_vec_t &train, const dce_lab::sample_vec_t &test,
               const dce_lab::param_dic_t &param_dict) {
    // todo
}

int XGB::Predict(const dce_lab::sample_vec_t &test, dce_lab::pred_res_t &result) const {
    // todo
}

int XGB::Save(const char *fname) const {
    // todo
}

int XGB::Load(const char *fname) {
    // todo
}

}   // namespace dce_lab
