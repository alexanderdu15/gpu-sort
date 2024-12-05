 #include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <fstream>
#include <iomanip>
#include <ctime>
#include "gpu_sorter.cuh"

using namespace std;

template<typename T>
void runTestForType(const string& type_name, uint32_t* sizes, int RUNS_PER_SIZE,
                   SortMethod* methods, int num_methods, ofstream& outfile) {
    for (int m = 0; m < num_methods; m++) {
        auto method = methods[m];
        string method_name = "SingleThread";
        if (method == SortMethod::PARALLEL_NAIVE) {
            method_name = "ParallelNaive";
        }
        else if (method == SortMethod::PARALLEL_SHARED_MEM) {
            method_name = "ParallelSharedMem";
        }
        cout << "\nTesting " << type_name << " with " << method_name << " sort" << endl;

        for (int s = 0; s < 7; s++) {  // assuming 7 sizes as in original array
            uint32_t size = sizes[s];
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
}

int main() {
    uint32_t sizes[] = {10, 100, 1000, 10000, 100000, 1000000, 10000000}; //TODO: compare performance between perfect powers of 2 vs non-powers
    const int RUNS_PER_SIZE = 3;
    SortMethod methods[] = {
        SortMethod::SINGLE_THREAD,
        SortMethod::PARALLEL_NAIVE,
        SortMethod::PARALLEL_SHARED_MEM
    };
    const int NUM_METHODS = sizeof(methods) / sizeof(methods[0]);
    
    time_t now = time(0);
    char timestamp[20];
    strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", localtime(&now));
    
    ofstream outfile("results/sorting_results_" + string(timestamp) + ".csv");
    outfile << "Type,Method,Size,Run,Correct,Total Time (ms),Kernel Time (ms)" << endl;

    // Run tests for each type
    runTestForType<uint32_t>("uint32", sizes, RUNS_PER_SIZE, methods, NUM_METHODS, outfile);
    runTestForType<int32_t>("int32", sizes, RUNS_PER_SIZE, methods, NUM_METHODS, outfile);
    runTestForType<uint64_t>("uint64", sizes, RUNS_PER_SIZE, methods, NUM_METHODS, outfile);
    runTestForType<int64_t>("int64", sizes, RUNS_PER_SIZE, methods, NUM_METHODS, outfile);
    runTestForType<float>("float", sizes, RUNS_PER_SIZE, methods, NUM_METHODS, outfile);
    runTestForType<double>("double", sizes, RUNS_PER_SIZE, methods, NUM_METHODS, outfile);
    
    outfile.close();
    cout << "\nResults have been written to sorting_results_" << string(timestamp) << ".csv" << endl;
    
    return 0;
}