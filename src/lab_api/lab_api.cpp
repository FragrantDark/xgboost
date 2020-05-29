//
// Created by zhangqi on 2020-05-28.
//
#include <vector>

#include <lab_api.h>
#include <xgboost/c_api.h>

#include "../data/simple_dmatrix.h"

namespace dce_lab {

/**
 * convert sample_vec_t instance to SimpleDMatrix
 * @param svec
 * @param sdmtx
 * @return  0 if success, <0 else
 *
 */
int SampleVec2SimpleDMatrix(const sample_vec_t& svec, xgboost::data::SimpleDMatrix& sdmtx, int n_col) {

    int non_zero_cnt = 0;
    std::vector<bst_row_t> offset_vec;
    std::vector<xgboost::Entry> data_vec;
    std::vector<bst_float> labels_vec;
    std::vector<bst_float> weights_vec;

    offset_vec.push_back(0);
    for (auto spl=svec.cbegin(); spl!=svec.cend(); ++spl) {
        labels_vec.emplace_back(spl->label);
        weights_vec.emplace_back(spl->weight);
        non_zero_cnt += spl->features.size();

        for (auto ety=spl->features.cbegin(); ety!=spl->features.cend(); ++ety) {
            data_vec.emplace_back(ety->first, ety->second);
        }
        offset_vec.emplace_back(data_vec.size());
    }

    // 1. SparsePage
    xgboost::SparsePage page;

    page.offset = xgboost::HostDeviceVector(offset_vec);
    page.data = xgboost::HostDeviceVector(data_vec);

    sdmtx.SetSparsePage(page);

    // 2. MetaInfo
    xgboost::MetaInfo info;

    info.num_row_ = svec.size();
    info.num_col_ = n_col;
    info.num_nonzero_ = non_zero_cnt;
    info.labels_ = xgboost::HostDeviceVector(labels_vec);
    info.weights_ = xgboost::HostDeviceVector(weights_vec);

    sdmtx.SetMetaInfo(info);

    return 0;
}

int XGB::Train(const dce_lab::sample_vec_t &train, const dce_lab::sample_vec_t &test,
               const dce_lab::param_dic_t &param_dict) {

    xgboost::data::SimpleDMatrix dtrain;
    SampleVec2SimpleDMatrix(train, dtrain, n_col_);

    xgboost::data::SimpleDMatrix dtest;
    SampleVec2SimpleDMatrix(test, dtest, n_col_);

    DMatrixHandle eval_dmats[2] = {dtrain, dtest};

    safe_xgboost(XGBoosterCreate(eval_dmats, 2, &booster));

    // set params
    for (auto it=param_dict.begin(); it!=param_dict.end(); ++it) {
        safe_xgboost(XGBoosterSetParam(booster, it->first, it->second);
    }

    // train and evaluate for 10 iters
    // todo
    int n_trees = 10;
    const char* eval_names[2] = {"train", "test"};
    const char* eval_result = NULL;
    for (int i = 0; i < n_trees; ++i) {
        safe_xgboost(XGBoosterUpdateOneIter(booster, i, dtrain));
        safe_xgboost(XGBoosterEvalOneIter(booster, i, eval_dmats, eval_names, 2, &eval_result));
        printf("%s\n", eval_result);
    }

    safe_xgboost(XGDMatrixFree(dtrain));
    safe_xgboost(XGDMatrixFree(dtest));

    return 0;
}

int XGB::Predict(const dce_lab::sample_vec_t &test, dce_lab::pred_res_t &result) const {

    bst_ulong out_len = 0;
    const float* out_result = NULL;

    xgboost::data::SimpleDMatrix dtest;
    SampleVec2SimpleDMatrix(test, dtest);

    safe_xgboost(XGBoosterPredict(booster, dtest, 0, 0, 0, &out_len, &out_result));

    result.clear()
    for (int i=0; i<out_len; ++i) {
        result.emplace_back(out_result[i]);
    }

    safe_xgboost(XGDMatrixFree(dtest));

    return 0;
}

int XGB::Save(const char *fname) const {
    safe_xgboost(XGBoosterSaveModel(booster, fname));

    return 0;
}

int XGB::Load(const char *fname) {
    safe_xgboost(XGBoosterLoadModel(booster, fname));

    return 0;
}

}   // namespace dce_lab
