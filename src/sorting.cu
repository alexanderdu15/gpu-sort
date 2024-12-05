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
        string method_name = "";
        if (method == SortMethod::SINGLE_THREAD)
            method_name = "SingleThread";
        else if (method == SortMethod::PARALLEL_NAIVE)
            method_name = "ParallelNaive";
        else if (method == SortMethod::PARALLEL_SHARED)
            method_name = "ParallelShared";

        cout << "\nTesting " << type_name << " with " << method_name << " sort" << endl;

        for (int s = 0; s < 16; s++) {  // assuming 8 sizes as in original array
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
    uint32_t sizes[] = {8, 10, 100, 128, 1000, 1024, 10000, 16384, 100000, 131072, 1000000, 1048576, 10000000, 16777216, 100000000, 134217728};
    const int RUNS_PER_SIZE = 3;
    SortMethod methods[] = {
        // SortMethod::SINGLE_THREAD, // too slow to test!
        SortMethod::PARALLEL_NAIVE,
        SortMethod::PARALLEL_SHARED
    };
    const int NUM_METHODS = sizeof(methods) / sizeof(methods[0]);
    
    time_t now = time(0);
    char timestamp[20];
    strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", localtime(&now));
    
    ofstream outfile("results/sorting_results_" + string(timestamp) + ".csv");
    outfile << "Type,Method,Size,Run,Correct,Total Time (ms),Kernel Time (ms)" << endl;

    // Run tests for each type
    // Can't run all at once -> Shared memory can't have same name and diff types
    //runTestForType<uint32_t>("uint32", sizes, RUNS_PER_SIZE, methods, NUM_METHODS, outfile);
    // runTestForType<int32_t>("int32", sizes, RUNS_PER_SIZE, methods, NUM_METHODS, outfile);
    //runTestForType<uint64_t>("uint64", sizes, RUNS_PER_SIZE, methods, NUM_METHODS, outfile);
    //runTestForType<int64_t>("int64", sizes, RUNS_PER_SIZE, methods, NUM_METHODS, outfile);
    runTestForType<float>("float", sizes, RUNS_PER_SIZE, methods, NUM_METHODS, outfile);
    //runTestForType<double>("double", sizes, RUNS_PER_SIZE, methods, NUM_METHODS, outfile);
    
    outfile.close();
    cout << "\nResults have been written to sorting_results_" << string(timestamp) << ".csv" << endl;
    
    return 0;
}