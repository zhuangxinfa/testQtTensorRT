#ifndef SAMPLEUFFMNIST_H
#define SAMPLEUFFMNIST_H

#endif // SAMPLEUFFMNIST_H

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

const std::string gSampleName = "TensorRT.sample_uff_mnist";

//!
//! \brief  The SampleUffMNIST class implements the UffMNIST sample
//!
//! \details It creates the network using a Uff model
//!
class SampleUffMNIST
{
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    SampleUffMNIST(const samplesCommon::UffSampleParams& params)
        : mParams(params)
    {
    }

    //!
    //! \brief Builds the network engine
    //!
    bool build();

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool infer();

    //!
    //! \brief Used to clean up any state created in the sample class
    //!
    bool teardown();

private:
    //!
    //! \brief Parses a Uff model for MNIST and creates a TensorRT network
    //!
    void constructNetwork(SampleUniquePtr<nvuffparser::IUffParser>& parser,
                          SampleUniquePtr<nvinfer1::INetworkDefinition>& network);

    //!
    //! \brief Reads the input and mean data, preprocesses, and stores the result
    //!        in a managed buffer
    //!
    bool processInput(const samplesCommon::BufferManager& buffers,
                      const std::string& inputTensorName, int inputFileIdx) const;

    //!
    //! \brief Verifies that the output is correct and prints it
    //!
    bool verifyOutput(const samplesCommon::BufferManager& buffers,
                      const std::string& outputTensorName,
                      int groundTruthDigit) const;

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine{nullptr}; //!< The TensorRT engine used to run the network

    samplesCommon::UffSampleParams mParams;

    nvinfer1::Dims mInputDims;
    const int kDIGITS{10};
};
