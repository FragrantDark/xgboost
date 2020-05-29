//
// Created by zhangqi on 2020-05-28.
//
#include <vector>
#include <iostream>
#include <string>
#include <sstream>

#include <lab_api.h>
#include <xgboost/c_api.h>

#include "../data/simple_dmatrix.h"

#define safe_xgboost(call) {                                            \
int err = (call);                                                       \
if (err != 0) {                                                         \
  fprintf(stderr, "%s:%d: error in %s: %s\n", __FILE__, __LINE__, #call, XGBGetLastError()); \
  exit(1);                                                              \
}                                                                       \
}

namespace dce_lab {

template<typename T> std::string v2s(const std::vector<T>& v) {
    std::stringstream ss;
    for (auto it=v.cbegin(); it!=v.cend(); ++it) {
        ss << (*it) << ",";
    }
    std::string str;
    ss >> str;
    return str;
}

/**
 * convert sample_vec_t instance to SimpleDMatrix
 * @param svec
 * @param sdmtx
 * @return  0 if success, <0 else
 *
 */
void TestDMatrix(const xgboost::MetaInfo& info, const xgboost::SparsePage& spage) {
    std::cerr << "MetaInfo:" << std::endl
              << "\tnum_row_\t" << info.num_row_ << std::endl
              << "\tnum_col_\t" << info.num_col_ << std::endl
              << "\tnum_nonzero_\t" << info.num_nonzero_ << std::endl
              << "\tlabels_\t" << info.labels_.HostVector().size() << "\t" << v2s(info.labels_.HostVector()) << std::endl
              << "\tgroup_ptr_\t" << info.group_ptr_.size() << "\t" << v2s(info.group_ptr_) << std::endl
              << "\tweights_\t" << info.weights_.HostVector().size() << "\t" << v2s(info.weights_.HostVector()) << std::endl
              << "\tbase_margin_\t" << info.base_margin_.HostVector().size() << "\t" << v2s(info.base_margin_.HostVector()) << std::endl
              << "\tlabels_lower_bound_\t" << info.labels_lower_bound_.HostVector().size() << "\t" << v2s(info.labels_lower_bound_.HostVector()) << std::endl
              << "\tlabels_upper_bound_\t" << info.labels_upper_bound_.HostVector().size() << "\t" << v2s(info.labels_upper_bound_.HostVector()) << std::endl
              << std::endl;

    std::cerr << "SparsePage:" << std::endl
              << "\toffset\t" << spage.offset.HostVector().size() << "\t" << v2s(spage.offset.HostVector()) << std::endl
              << "\tdata\t" << spage.data.HostVector().size() << "\t";

    const std::vector<xgboost::Entry>& v = spage.data.HostVector();
    for (auto it=v.cbegin(); it!=v.cend(); ++it) {
        std::cerr << it->index << ":" << it->fvalue << ",";
    }
    std::cerr << std::endl;

    std::cerr << "\tbase_rowid\t" << spage.base_rowid << std::endl << std::endl;
}


void TestDMatrix(xgboost::data::SimpleDMatrix& dmtx) {
    const xgboost::MetaInfo& info = dmtx.Info();
    const xgboost::SparsePage& spage = dmtx.GetSparsePage();

    TestDMatrix(info, spage);
}

void TestDMatrix(DMatrixHandle dmtx) {
    const xgboost::MetaInfo& info = ((xgboost::data::SimpleDMatrix*)dmtx)->Info();
    const xgboost::SparsePage& spage = ((xgboost::data::SimpleDMatrix*)dmtx)->GetSparsePage();

    TestDMatrix(info, spage);
}

int SampleVec2SimpleDMatrix(const sample_vec_t& svec, xgboost::data::SimpleDMatrix& sdmtx, int n_col) {

    int non_zero_cnt = 0;
    std::vector<xgboost::bst_row_t> offset_vec;
    std::vector<xgboost::Entry> data_vec;
    std::vector<xgboost::bst_float> labels_vec;
    std::vector<xgboost::bst_float> weights_vec;

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

    page.offset = xgboost::HostDeviceVector<xgboost::bst_row_t>(offset_vec);
    page.data = xgboost::HostDeviceVector<xgboost::Entry>(data_vec);
    page.base_rowid = 0;

    sdmtx.SetSparsePage((xgboost::SparsePage&&)page);

    // 2. MetaInfo
    xgboost::MetaInfo info;

    info.num_row_ = svec.size();
    info.num_col_ = n_col;
    info.num_nonzero_ = non_zero_cnt;
    info.labels_ = xgboost::HostDeviceVector<xgboost::bst_float>(labels_vec);
    info.weights_ = xgboost::HostDeviceVector<xgboost::bst_float>(weights_vec);

    sdmtx.SetMetaInfo(info);

    return 0;
}

XGB::~XGB() { safe_xgboost(XGBoosterFree(booster)); }

int XGB::Train(const dce_lab::sample_vec_t &train, const dce_lab::sample_vec_t &test,
               const dce_lab::param_dic_t &param_dict) {

    xgboost::data::SimpleDMatrix dtrain;
    SampleVec2SimpleDMatrix(train, dtrain, n_col_);

    xgboost::data::SimpleDMatrix dtest;
    SampleVec2SimpleDMatrix(test, dtest, n_col_);

    DMatrixHandle dtrain1 = new std::shared_ptr<xgboost::DMatrix>(&dtrain);
    DMatrixHandle dtest1 = new std::shared_ptr<xgboost::DMatrix>(&dtest);

    DMatrixHandle eval_dmats[2] = {dtrain1, dtest1};

    safe_xgboost(XGBoosterCreate(eval_dmats, 2, &booster));

    // set params
    for (auto it=param_dict.begin(); it!=param_dict.end(); ++it) {
        safe_xgboost(XGBoosterSetParam(booster, it->first.c_str(), it->second.c_str()));
    }

    // train and evaluate for 10 iters
    // todo
    int n_trees = 10;
    const char* eval_names[2] = {"train", "test"};
    const char* eval_result = NULL;
    for (int i = 0; i < n_trees; ++i) {
        //safe_xgboost(XGBoosterUpdateOneIter(booster, i, &dtrain));
        safe_xgboost(XGBoosterUpdateOneIter(booster, i, dtrain1));
        safe_xgboost(XGBoosterEvalOneIter(booster, i, eval_dmats, eval_names, 2, &eval_result));
        printf("%s\n", eval_result);
    }

    return 0;
}

int XGB::Predict(const dce_lab::sample_vec_t &test, dce_lab::pred_res_t &result) const {


    bst_ulong out_len = 0;
    const float* out_result = NULL;

    xgboost::data::SimpleDMatrix dtest;
    SampleVec2SimpleDMatrix(test, dtest, n_col_);

    DMatrixHandle dtest1 = new std::shared_ptr<xgboost::DMatrix>(&dtest);

    safe_xgboost(XGBoosterPredict(booster, dtest1, 0, 0, 0, &out_len, &out_result));

    result.clear();
    for (int i=0; i<out_len; ++i) {
        result.emplace_back(out_result[i]);
    }

    return 0;
}

int XGB::Save(const char *fname) const {
    safe_xgboost(XGBoosterSaveModel(booster, fname));

    return 0;
}

int XGB::Load(const char *fname, const sample_vec_t& samples) {

    xgboost::data::SimpleDMatrix samples_dmtx;
    SampleVec2SimpleDMatrix(samples, samples_dmtx, n_col_);

    DMatrixHandle dmtx = new std::shared_ptr<xgboost::DMatrix>(&samples_dmtx);

    DMatrixHandle eval_dmats[2] = {dmtx, dmtx};

    safe_xgboost(XGBoosterCreate(eval_dmats, 2, &booster));

    safe_xgboost(XGBoosterLoadModel(booster, fname));

    return 0;
}

}   // namespace dce_lab
