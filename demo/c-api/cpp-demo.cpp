//
// Created by zhangqi on 2020-05-27.
//
#include <iostream>
#include <sstream>
#include <string>


#include "../../src/data/simple_dmatrix.h"
#include "../../include/xgboost/data.h"


using namespace xgboost;    // NOLINT
using namespace std;

template<typename T> string v2s(const vector<T>& v) {
    stringstream ss;
    for (auto it=v.cbegin(); it!=v.cend(); ++it) {
        ss << (*it) << ",";
    }
    string str;
    ss >> str;
    return str;
}

int main(int argc, char** argv) {
    // added by zhangqi01
    xgboost::DMatrix* dmat = xgboost::DMatrix::Load("../data/agaricus.txt.test", true, false);
    const xgboost::MetaInfo& info = ((xgboost::data::SimpleDMatrix*)dmat)->Info();
    const xgboost::SparsePage& spage = ((xgboost::data::SimpleDMatrix*)dmat)->GetSparsePage();

    cerr << "MetaInfo:" << endl
        << "\tnum_row_\t" << info.num_row_ << endl
        << "\tnum_col_\t" << info.num_col_ << endl
        << "\tnum_nonzero_\t" << info.num_nonzero_ << endl
        << "\tlabels_\t" << info.labels_.HostVector().size() << "\t" << v2s(info.labels_.HostVector()) << endl
        << "\tgroup_ptr_\t" << info.group_ptr_.size() << "\t" << v2s(info.group_ptr_) << endl
        << "\tweights_\t" << info.weights_.HostVector().size() << "\t" << v2s(info.weights_.HostVector()) << endl
        << "\tbase_margin_\t" << info.base_margin_.HostVector().size() << "\t" << v2s(info.base_margin_.HostVector()) << endl
        << "\tlabels_lower_bound_\t" << info.labels_lower_bound_.HostVector().size() << "\t" << v2s(info.labels_lower_bound_.HostVector()) << endl
        << "\tlabels_upper_bound_\t" << info.labels_upper_bound_.HostVector().size() << "\t" << v2s(info.labels_upper_bound_.HostVector()) << endl
        << endl;

    cerr << "SparsePage:" << endl
        << "\toffset\t" << spage.offset.HostVector().size() << "\t" << v2s(spage.offset.HostVector()) << endl
        << "\tdata\t" << spage.data.HostVector().size() << "\t";

    const vector<Entry>& v = spage.data.HostVector();
    for (auto it=v.cbegin(); it!=v.cend(); ++it) {
        cerr << it->index << ":" << it->fvalue << ",";
    }
    cerr << endl << endl;

}
