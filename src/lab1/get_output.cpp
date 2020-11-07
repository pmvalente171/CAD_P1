//
// Created by pm_valente on 07/11/20.
//

#include "get_output.h"

void get_output::save_values_by_iteration() {

}

get_output::get_output(const string& file_name) {
    outfile.open(file_name);
}

get_output::~get_output() {
    outfile.close();
}
