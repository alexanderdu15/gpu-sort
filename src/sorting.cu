#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <fstream>
#include <iomanip>
#include <ctime>
#include "gpu_sorter.cuh"

using namespace std;

int main() {
    uint32_t sizes[] = {10, 100, 1000, 10000, 100000, 1000000, 10000000}; //TODO: compare performance between perfect powers of 2 vs non-powers
    const int RUNS_PER_SIZE = 3;
    SortMethod methods[] = {
        SortMethod::SINGLE_THREAD,
        SortMethod::PARALLEL_NAIVE
    };
    
    time_t now = time(0);
    char timestamp[20];
    strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", localtime(&now));
    
    ofstream outfile("results/sorting_results_" + string(timestamp) + ".csv");
    outfile << "Type,Method,Size,Run,Correct,Total Time (ms),Kernel Time (ms)" << endl;

    auto runTestForType = [&](auto type, const string& type_name) {
        using T = decltype(type);
        
        for (auto method : methods) {
            string method_name = (method == SortMethod::PARALLEL_NAIVE) ? "ParallelNaive" : "SingleThread";
            cout << "\nTesting " << type_name << " with " << method_name << " sort" << endl;
            
            for (uint32_t size : sizes) {
                cout << "Array size " << size << "..." << endl;
                
                for (int run = 0; run < RUNS_PER_SIZE; run++) {
                    GPUSorter<T> sorter(size, method);
                    sorter.generateRandomData();
                    sorter.sort();
                    bool is_sorted = sorter.checkResult();
                    
                    outfile << type_name << ","
                           << method_name << ","
                           << size << ","
                           << run + 1 << ","
                           << (is_sorted ? "true" : "false") << ","
                           << fixed << setprecision(5) << sorter.getTotalTime() << ","
                           << fixed << setprecision(5) << sorter.getKernelTime() << endl;
                    
                    cout << "Run " << (run + 1) << " complete - "
                         << (is_sorted ? "Correct" : "Incorrect") << endl;
                }
            }
        }
    };
    
    runTestForType(uint32_t{}, "uint32");
    runTestForType(int32_t{}, "int32");
    runTestForType(uint64_t{}, "uint64");
    runTestForType(int64_t{}, "int64");
    runTestForType(float{}, "float");
    runTestForType(double{}, "double");
    
    outfile.close();
    cout << "\nResults have been written to sorting_results_" << string(timestamp) << ".csv" << endl;
    
    return 0;
}