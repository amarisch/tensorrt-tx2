#include <algorithm>
#include "tensorNet.h"



void TensorNet::caffeToTRTModel(const std::string& deployFile,
                                const std::string& modelFile,
                                const std::vector<std::string>& outputs,
                                unsigned int maxBatchSize)
{
    IBuilder* builder = createInferBuilder(gLogger);
    INetworkDefinition* network = builder->createNetwork();

    ICaffeParser* parser = createCaffeParser();
    parser->setPluginFactory(&pluginFactory);

    bool useFp16 = builder->platformHasFastFp16();
    DataType modelDataType = useFp16 ? DataType::kHALF : DataType::kFLOAT;

    const IBlobNameToTensor *blobNameToTensor =	parser->parse(deployFile.c_str(),
                                                              modelFile.c_str(),
                                                              *network,
                                                              modelDataType);

    assert(blobNameToTensor != nullptr);

    // specify which tensors are outputs
    for (auto& s : outputs) network->markOutput(*blobNameToTensor->find(s.c_str()));

    // Build the engine
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(16 << 20);

    if(useFp16) builder->setHalf2Mode(true);

    ICudaEngine* engine = builder->buildCudaEngine(*network);
    assert(engine);

    network->destroy();
    parser->destroy();

    // serialize the engine, then close everything down
    gieModelStream = engine->serialize();
    engine->destroy();
    builder->destroy();
    pluginFactory.destroyPlugin();
    shutdownProtobufLibrary();
}

void TensorNet::createInference()
{
    infer = createInferRuntime(gLogger);
    engine = infer->deserializeCudaEngine(gieModelStream->data(), gieModelStream->size(), &pluginFactory);

    printf("Bindings after deserializing:\n"); 
    for (int bi = 0; bi < engine->getNbBindings(); bi++) {
        if (engine->bindingIsInput(bi) == true) printf("Binding %d (%s): Input.\n",  bi, engine->getBindingName(bi));
        else printf("Binding %d (%s): Output.\n", bi, engine->getBindingName(bi));
    }
}

void TensorNet::imageInference(void** buffers, int nbBuffer, int batchSize)
{
    assert(engine->getNbBindings()==nbBuffer);

    IExecutionContext* context = engine->createExecutionContext();
    context->setProfiler(&gProfiler);

    float* input = (float *)buffers[0];
    float* output = (float *)buffers[1];

    void* localbuffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // note that indices are guaranteed to be less than IEngine::getNbBindings()
    int inputIndex = 0;
    int  outputIndex = 1;

    int INPUT_H = 256;
    int INPUT_W = 256;
    int OUTPUT_SIZE = 2;

    // create GPU buffers and a stream
    CHECK(cudaMalloc(&localbuffers[inputIndex], batchSize * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&localbuffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
    CHECK(cudaMemcpyAsync(localbuffers[inputIndex], input, batchSize * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context->enqueue(batchSize, localbuffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, localbuffers[outputIndex], batchSize * OUTPUT_SIZE*sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // release the stream and the buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(localbuffers[inputIndex]));
    CHECK(cudaFree(localbuffers[outputIndex]));
    
    //context->execute(batchSize, buffers);
    context->destroy();

    //int argMax = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
    //printf("Inference result: %d\n", argMax);
    printf("Inference result 0: %f\n", (double )output[0]);
    printf("Inference result 1: %f\n", (double )output[1]);
	
}

void TensorNet::timeInference(int iteration, int batchSize)
{
    int inputIdx = 0;
    size_t inputSize = 0;

    void* buffers[engine->getNbBindings()];

    for (int b = 0; b < engine->getNbBindings(); b++) {
        DimsCHW dims = static_cast<DimsCHW&&>(engine->getBindingDimensions(b));
        size_t size = batchSize * dims.c() * dims.h() * dims.w() * sizeof(float);
        CHECK(cudaMalloc(&buffers[b], size));

        if(engine->bindingIsInput(b) == true)
        {
            inputIdx = b;
            inputSize = size;
        }
    }

    IExecutionContext* context = engine->createExecutionContext();
    context->setProfiler(&gProfiler);

    CHECK(cudaMemset(buffers[inputIdx], 0, inputSize));

    for (int i = 0; i < iteration;i++) context->execute(batchSize, buffers);

    context->destroy();
    for (int b = 0; b < engine->getNbBindings(); b++) CHECK(cudaFree(buffers[b]));
}

DimsCHW TensorNet::getTensorDims(const char* name)
{
    for (int b = 0; b < engine->getNbBindings(); b++) {
        if( !strcmp(name, engine->getBindingName(b)) )
            return static_cast<DimsCHW&&>(engine->getBindingDimensions(b));
    }
    return DimsCHW{0,0,0};
}

void TensorNet::printTimes(int iteration)
{
    gProfiler.printLayerTimes(iteration);
}

void TensorNet::destroy()
{
    pluginFactory.destroyPlugin();
    engine->destroy();
    infer->destroy();
}
