#include "common/argsParser.h"
#include "common/buffers.h"
#include "common/common.h"
#include "common/logger.h"

#include "NvInfer.h"
#include "NvUffParser.h"
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <vector>
#include"sampleuffmnist.h"

//!
//! \brief Prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout << "Usage: ./sample_uff_mnist [-h or --help] [-d or "
                 "--datadir=<path to data directory>] [--useDLACore=<int>]\n";
    std::cout << "--help          Display help information\n";
    std::cout << "--datadir       Specify path to a data directory, overriding "
                 "the default. This option can be used multiple times to add "
                 "multiple directories. If no data directories are given, the "
                 "default is to use (data/samples/mnist/, data/mnist/)"
              << std::endl;
    std::cout << "--useDLACore=N  Specify a DLA engine for layers that support "
                 "DLA. Value can range from 0 to n-1, where n is the number of "
                 "DLA engines on the platform."
              << std::endl;
    std::cout << "--int8          Run in Int8 mode.\n";
    std::cout << "--fp16          Run in FP16 mode." << std::endl;
}


//!
//! \brief Initializes members of the params struct
//!        using the command line args
//!
samplesCommon::UffSampleParams
initializeSampleParams(const samplesCommon::Args& args)
{
    samplesCommon::UffSampleParams params;
    if (args.dataDirs.empty()) //!< Use default directories if user hasn't provided paths
    {
        params.dataDirs.push_back("/usr/src/tensorrt/data/mnist/");
        params.dataDirs.push_back("/usr/src/tensorrt/samples/mnist/");
    }
    else //!< Use the data directory provided by the user
    {
        params.dataDirs = args.dataDirs;
    }

    params.uffFileName = locateFile("lenet5.uff", params.dataDirs);
    params.inputTensorNames.push_back("in");
    params.batchSize = 1;
    params.outputTensorNames.push_back("out");
    params.dlaCore = args.useDLACore;
    params.int8 = args.runInInt8;
    params.fp16 = args.runInFp16;

    return params;
}


int main(int argc, char** argv)
{
    samplesCommon::Args args;
    bool argsOK = samplesCommon::parseArgs(args, argc, argv);
    if (!argsOK)
    {
        gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }
    auto sampleTest = gLogger.defineTest(gSampleName, argc, argv);

    gLogger.reportTestStart(sampleTest);

    samplesCommon::UffSampleParams params = initializeSampleParams(args);

    SampleUffMNIST sample(params);
    gLogInfo << "Building and running a GPU inference engine for Uff MNIST"
             << std::endl;
    //解析模型构建network
    if (!sample.build())
    {
        return gLogger.reportFail(sampleTest);
    }
    if (!sample.infer())
    {
        return gLogger.reportFail(sampleTest);
    }
    if (!sample.teardown())
    {
        return gLogger.reportFail(sampleTest);
    }

    return gLogger.reportPass(sampleTest);
}
