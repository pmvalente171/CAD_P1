//
// Created by pm_valente on 07/11/20.
//

#include <fstream>
#include <iostream>
using namespace std;

#ifndef CADLABS_GPU_GET_OUTPUT_H
#define CADLABS_GPU_GET_OUTPUT_H


class get_output {
public:
    void save_values_by_iteration();
    explicit get_output(const string& file_name);
    ~get_output();

private:
    ofstream outfile;

};


#endif //CADLABS_GPU_GET_OUTPUT_H
