#!/bin/bash
cmd="c++ -O3 -I../../include -I../../dmlc-core/include -I../../rabit/include -L../../lib -o cpp-demo cpp-demo.cpp -lxgboost -Wl,-rpath,../../lib -std=c++14"
echo $cmd
$cmd
