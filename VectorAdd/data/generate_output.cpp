#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <ctime>

/* Get random number from 0 to 4 */
float generateRandomFloat() {
    return static_cast<float>(rand()) / RAND_MAX * 4.0f; 
}

void generateFile(const std::string& filename, size_t length) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    file << length << "\n";
    
    for (size_t i = 0; i < length; ++i) {
        file << generateRandomFloat() << "\n";
    }
    
    file.close();
}

void generateOutputFile(const std::string& input1File, const std::string& input2File, const std::string& outputFile) {
    std::ifstream file1(input1File);
    std::ifstream file2(input2File);
    std::ofstream outFile(outputFile);

    if (!file1.is_open() || !file2.is_open() || !outFile.is_open()) {
        std::cerr << "Failed to open input or output file" << std::endl;
        exit(EXIT_FAILURE);
    }

    size_t length1, length2;
    file1 >> length1;
    file2 >> length2;

    if (length1 != length2) {
        std::cerr << "Input files have different lengths" << std::endl;
        exit(EXIT_FAILURE);
    }

    outFile << length1 << "\n";

    std::vector<float> data1(length1);
    std::vector<float> data2(length2);

    for (size_t i = 0; i < length1; ++i) {
        file1 >> data1[i];
    }
    
    for (size_t i = 0; i < length2; ++i) {
        file2 >> data2[i];
    }
    // CPU kernel here
    for (size_t i = 0; i < length1; ++i) {
        outFile << data1[i] + data2[i] << "\n";
    }

    file1.close();
    file2.close();
    outFile.close();
}

int main() {
    srand(static_cast<unsigned>(time(0))); 

    size_t length = 940312518;

    generateFile("input0.raw", length);
    generateFile("input1.raw", length);

    generateOutputFile("input0.raw", "input1.raw", "output.raw");

    std::cout << "Files generated successfully." << std::endl;
    return 0;
}
