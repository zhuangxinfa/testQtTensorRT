#include"sampleuffmnist.h"
//!
//! sampleUffMNIST.cpp
//! This file contains the implementation of the Uff MNIST sample.
//! It creates the network using the MNIST model converted to uff.
//!
//! It can be run with the following command line:
//! Command: ./sample_uff_mnist [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]
//!


//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates the MNIST network by parsing the Uff model
//!          and builds the engine that will be used to run MNIST (mEngine)
//!
//! \return Returns true if the engine was created successfully and false otherwise
//!
bool SampleUffMNIST::build()
{
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(
        nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetwork());
    if (!network)
    {
        return false;
    }
    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }
    auto parser = SampleUniquePtr<nvuffparser::IUffParser>(nvuffparser::createUffParser());
    if (!parser)
    {
        return false;
    }
    constructNetwork(parser, network);
    builder->setMaxBatchSize(mParams.batchSize);
    config->setMaxWorkspaceSize(16_MiB);
    config->setFlag(BuilderFlag::kGPU_FALLBACK);
    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    if (mParams.int8)
    {
        config->setFlag(BuilderFlag::kINT8);
    }

    samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());

    if (!mEngine)
    {
        return false;
    }
    assert(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    assert(mInputDims.nbDims == 3);

    return true;
}

//!
//! \brief Uses a Uff parser to create the MNIST Network and marks the output layers
//!
//! \param network Pointer to the network that will be populated with the MNIST network
//!
//! \param builder Pointer to the engine builder
//!
void SampleUffMNIST::constructNetwork(
    SampleUniquePtr<nvuffparser::IUffParser>& parser,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network)
{
    // There should only be one input and one output tensor
    assert(mParams.inputTensorNames.size() == 1);
    assert(mParams.outputTensorNames.size() == 1);

    // Register tensorflow input
    parser->registerInput(mParams.inputTensorNames[0].c_str(),
                          nvinfer1::Dims3(1, 28, 28),
                          nvuffparser::UffInputOrder::kNCHW);
    parser->registerOutput(mParams.outputTensorNames[0].c_str());

    parser->parse(mParams.uffFileName.c_str(), *network, nvinfer1::DataType::kFLOAT);

    if (mParams.int8)
    {
        samplesCommon::setAllTensorScales(network.get(), 127.0f, 127.0f);
    }
}

//!
//! \brief Reads the input data, preprocesses, and stores the result in a managed buffer
//!
bool SampleUffMNIST::processInput(const samplesCommon::BufferManager& buffers,
                                  const std::string& inputTensorName,
                                  int inputFileIdx) const
{
    const int inputH = mInputDims.d[1];
    const int inputW = mInputDims.d[2];

    std::vector<uint8_t> fileData(inputH * inputW);
    readPGMFile(
        locateFile(std::to_string(inputFileIdx) + ".pgm", mParams.dataDirs),
        fileData.data(), inputH, inputW);

    // Print ASCII representation of digit
    gLogInfo << "Input:\n";
    for (int i = 0; i < inputH * inputW; i++)
    {
        gLogInfo << (" .:-=+*#%@"[fileData[i] / 26])
                 << (((i + 1) % inputW) ? "" : "\n");
    }
    gLogInfo << std::endl;

    float* hostInputBuffer = static_cast<float*>(buffers.getHostBuffer(inputTensorName));

    for (int i = 0; i < inputH * inputW; i++)
    {
        hostInputBuffer[i] = 1.0 - float(fileData[i]) / 255.0;
    }
    return true;
}

//!
//! \brief Verifies that the inference output is correct
//!
bool SampleUffMNIST::verifyOutput(const samplesCommon::BufferManager& buffers,
                                  const std::string& outputTensorName,
                                  int groundTruthDigit) const
{
    const float* prob = static_cast<const float*>(buffers.getHostBuffer(outputTensorName));

    gLogInfo << "Output:\n";

    float val{0.0f};
    int idx{0};

    // Determine index with highest output value
    for (int i = 0; i < kDIGITS; i++)
    {
        if (val < prob[i])
        {
            val = prob[i];
            idx = i;
        }
    }

    // Print output values for each index
    for (int j = 0; j < kDIGITS; j++)
    {
        gLogInfo << j << "=> " << setw(10) << prob[j] << "\t : ";

        // Emphasize index with highest output value
        if (j == idx)
        {
            gLogInfo << "***";
        }
        gLogInfo << "\n";
    }

    gLogInfo << std::endl;
    return (idx == groundTruthDigit);
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample.
//!  It allocates the buffer, sets inputs, executes the engine, and verifies the output.
//!
bool SampleUffMNIST::infer()
{
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine, mParams.batchSize);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(
        mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    bool outputCorrect = true;
    float total = 0;

    // Try to infer each digit 0-9
    for (int digit = 0; digit < kDIGITS; digit++)
    {
        if (!processInput(buffers, mParams.inputTensorNames[0], digit))
        {
            return false;
        }
        // Copy data from host input buffers to device input buffers
        buffers.copyInputToDevice();

        const auto t_start = std::chrono::high_resolution_clock::now();

        // Execute the inference work
        if (!context->execute(mParams.batchSize,
                              buffers.getDeviceBindings().data()))
        {
            return false;
        }

        const auto t_end = std::chrono::high_resolution_clock::now();
        const float ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
        total += ms;

        // Copy data from device output buffers to host output buffers
        buffers.copyOutputToHost();

        // Check and print the output of the inference
        outputCorrect &= verifyOutput(buffers, mParams.outputTensorNames[0], digit);
    }

    total /= kDIGITS;

    gLogInfo << "Average over " << kDIGITS << " runs is " << total << " ms."
             << std::endl;

    return outputCorrect;
}

//!
//! \brief Used to clean up any state created in the sample class
//!
bool SampleUffMNIST::teardown()
{
    nvuffparser::shutdownProtobufLibrary();
    return true;
}

