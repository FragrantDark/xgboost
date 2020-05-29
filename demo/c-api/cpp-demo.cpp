//
// Created by zhangqi on 2020-05-27.
//
#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include <unordered_map>

#include <lab_api.h>

using namespace std;
using namespace dce_lab;

void File2SampleVec(const char* fname, sample_vec_t& svec) {
    //cerr << "File2SampleVec()" << endl;
    svec.clear();

    ifstream ifs(fname);
    string line;
    while (getline(ifs, line)) {
        float lbl = float(line[0] - '0');
        float wgt = 1.0;
        svec.emplace_back(lbl, wgt);

        //cerr << line << endl;

        int bpos = 2;
        int epos = line.find_first_of(' ', bpos);
        char c;
        int idx;
        float fval;
        while (epos != string::npos) {
            //int mpos = line.find_first_of(':', bpos);
            stringstream ss;
            ss << line.substr(bpos, epos - bpos);
            ss >> idx >> c >> fval;

            //cerr << "Feature: [" << idx << "|" << c << "|" << fval << "]" << endl;
            svec.back().AddFeature(idx, fval);

            bpos = epos + 1;
            epos = line.find_first_of(' ', bpos);
        }
        // last feature
        stringstream ss;
        ss << line.substr(bpos);
        ss >> idx >> c >> fval;

        //cerr << "Feature: [" << idx << "|" << c << "|" << fval << "]" << endl;
        svec.back().AddFeature(idx, fval);
    }

    ifs.close();
}

int main(int argc, char** argv) {
    sample_vec_t train;
    sample_vec_t test;
    File2SampleVec("../data/agaricus.txt.train", train);
    File2SampleVec("../data/agaricus.txt.test", test);

    unordered_map<string, string> pdic;
    pdic["tree_method"] = "hist";
    pdic["gpu_id"] = "-1";
    pdic["objective"] = "binary:logistic";
    pdic["min_child_weight"] = "1";
    pdic["gamma"] = "0.1";
    pdic["max_depth"] = "3";
    pdic["verbosity"] = "0";

    XGB xgb(127);
    xgb.Train(train, test, pdic);

    pred_res_t result;
    xgb.Predict(test, result);

    int n_print = 10;

    cerr << "Pred result:" << endl;
    for (int i=0; i< n_print; ++i) {
        cout << "[" << i << "]\t" << test[i].label << "\t" << result[i] << endl;
    }

    const char* model_file = "xgb.model";
    xgb.Save(model_file);

    XGB xgb2(127);
    xgb2.Load(model_file, test);
    xgb2.Predict(test, result);

    cerr << "Pred2 result:" << endl;
    for (int i=0; i< n_print; ++i) {
        cout << "[" << i << "]\t" << test[i].label << "\t" << result[i] << endl;
    }

    return 0;
}
