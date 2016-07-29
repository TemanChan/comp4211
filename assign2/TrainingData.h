#ifndef TRAININGDATA_H
#define TRAININGDATA_H

#include <vector>
#include <cstdlib>
#include <fstream>
#include <sstream>

class TrainingData {
public:
    TrainingData(const std::string filename);
    bool isEof() { return m_trainingDataFile.eof(); }
    void restart();
    void getTopology(std::vector<unsigned> &topology);
    void getNextSample(std::vector<double> &inputVals, std::vector<double> &targetVals);

private:
    std::ifstream m_trainingDataFile;
    std::vector<double> m_topology;
};

#endif//TRAININGDATA_H