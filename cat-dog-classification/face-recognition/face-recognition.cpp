#include <algorithm>
#include <fstream>
#include "gstCamera.h"
#include "glDisplay.h"
#include "glTexture.h"

#include "cudaNormalize.h"
#include "cudaOverlay.h"
#include "cudaFont.h"
#include "tensorNet.h"
#include "loadImage.h"


using namespace nvinfer1;
using namespace nvcaffeparser1;

static const int BATCH_SIZE = 1;
static const int TIMING_ITERATIONS = 100;

const char* model  = "/home/nvidia/cat-dog-classification/data/deploy.prototxt";
const char* weight = "/home/nvidia/cat-dog-classification/data/weights.caffemodel";
const char* label  = "/home/nvidia/cat-dog-classification/data/labels.txt";
const std::vector<std::string> directories{ "/home/nvidia/cat-dog-classification/data/" };
const char* test_image  = "5360.jpg";
const char* imgFilename = "/home/nvidia/cat-dog-classification/data/dog.jpg";

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "softmax";

static const int INPUT_H = 256;
static const int INPUT_W = 256;
static const int OUTPUT_SIZE = 2;


float* allocateMemory(DimsCHW dims, char* info)
{
    float* ptr;
    size_t size;
    std::cout << "Allocate memory: " << info << std::endl;
    size = BATCH_SIZE * dims.c() * dims.h() * dims.w();
    assert(!cudaMallocManaged(&ptr, size*sizeof(float)));
    return ptr;
}

// load label info
std::vector<std::string> loadLabelInfo(const char* filename)
{   
    assert(filename);
    std::vector<std::string> labelInfo;

    FILE* f = fopen(filename, "r");
    if( !f )
    {   
        printf("failed to open %s\n", filename);
        assert(0);
    }
    
    char str[512];
    while( fgets(str, 512, f) != NULL )
    {   
        const int syn = 9;  // length of synset prefix (in characters)
        const int len = strlen(str);
        
        if( len > syn && str[0] == 'n' && str[syn] == ' ' )
        {   
            str[syn]   = 0;
            str[len-1] = 0;
            
            const std::string b = (str + syn + 1);
            labelInfo.push_back(b);
        }
        else if( len > 0 )      // no 9-character synset prefix (i.e. from DIGITS snapshot)
        {   
            if( str[len-1] == '\n' ) str[len-1] = 0;
            labelInfo.push_back(str);
        }
    }
    fclose(f);
    return labelInfo;
}

std::string locateFile(const std::string& input, const std::vector<std::string> & directories)
{
    std::string file;
    const int MAX_DEPTH{10};
    bool found{false};
    for (auto &dir : directories)
    {
        file = dir + input;
        for (int i = 0; i < MAX_DEPTH && !found; i++)
        {
            std::ifstream checkFile(file);
            found = checkFile.is_open();
            if (found) break;
            file = "../" + file;
        }
        if (found) break;
        file.clear();
    }

    assert(!file.empty() && "Could not find a file due to it not existing in the data directory.");
    return file;
}

void readJPGFile(const std::string& fileName,  uint8_t *buffer, int inH, int inW)
{
    std::ifstream infile(fileName, std::ifstream::binary);
    assert(infile.is_open() && "Attempting to read from a file that is not open.");
    std::string magic, h, w, max;
    infile >> magic >> h >> w >> max;
    infile.seekg(1, infile.cur);
    infile.read(reinterpret_cast<char*>(buffer), inH*inW);
}

int main(int argc, char** argv)
{
    std::cout << "Building and running a GPU inference engine for " << model << ", N=" << BATCH_SIZE << "..." << std::endl;

    /* create networks */
    TensorNet tensorNet;
    std::vector<std::string> labelInfo = loadLabelInfo(label);
    tensorNet.caffeToTRTModel(model, weight, std::vector < std::string > { OUTPUT_BLOB_NAME }, BATCH_SIZE);
    tensorNet.createInference();

    //int8_t fileData[INPUT_H*INPUT_W];
    //readJPGFile(locateFile(test_image, directories), fileData, INPUT_H, INPUT_W);
	
    // load image from file on disk
    
    float* imgCPU    = NULL;
    float* imgCUDA   = NULL;
    int    imgWidth  = 0;
    int    imgHeight = 0;
		
    if( !loadImageRGBA(imgFilename, (float4**)&imgCPU, (float4**)&imgCUDA, &imgWidth, &imgHeight) )
    {
	printf("failed to load image '%s'\n", imgFilename);
	return 0;
    }

    //float confidence = 0.0f;

    float* fileData = imgCUDA;

    // parse the mean file and  subtract it from the image
    ICaffeParser* parser = createCaffeParser();
    IBinaryProtoBlob* meanBlob = parser->parseBinaryProto(locateFile("mean.binaryproto", directories).c_str());
    parser->destroy();

    const float *meanData = reinterpret_cast<const float*>(meanBlob->getData());

    float data[INPUT_H*INPUT_W];
    for (int i = 0; i < INPUT_H*INPUT_W; i++)
        data[i] = float(fileData[i])-meanData[i];

    meanBlob->destroy();
    // end parsing mean

    // run inference
    float output[OUTPUT_SIZE];
    void* buffers[] = {data, output};
    tensorNet.imageInference(buffers, 2, BATCH_SIZE);

    /* destory */
    tensorNet.destroy();
    tensorNet.printTimes(TIMING_ITERATIONS);

    std::cout << "Done." << std::endl;
    return 0;
}
