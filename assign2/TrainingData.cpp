#include <limits>
#include "TrainingData.h"
using namespace std;

TrainingData::TrainingData(const string filename)
{
    m_trainingDataFile.open(filename.c_str());

    string line;
    string label;

    getline(m_trainingDataFile, line);
    stringstream ss(line);
    ss >> label;
    if (this->isEof() || label.compare("topology:") != 0) {
        abort();
    }

    while (!ss.eof()) {
        unsigned n;
        ss >> n;
        m_topology.push_back(n);
    }
}

void TrainingData::restart()
{
    m_trainingDataFile.seekg(0);
    m_trainingDataFile.ignore(numeric_limits<streamsize>::max(), '\n');
}

void TrainingData::getTopology(vector<unsigned> &topology)
{
    topology.clear();
    topology.reserve(m_topology.size());
    for(int n: m_topology){
        topology.push_back(n);
    }
}

void TrainingData::getNextSample(vector<double> &inputVals, vector<double> &targetVals)
{
    inputVals.clear();
    targetVals.clear();

    string line;
    string label;
    getline(m_trainingDataFile, line);
    stringstream inss(line);
    getline(m_trainingDataFile, line);
    stringstream outss(line);

    inss >> label;
    if (label.compare("in:") == 0) {
        double oneValue;
        while (inss >> oneValue) {
            inputVals.push_back(oneValue);
        }
    }

    outss >> label;
    if (label.compare("out:") == 0) {
        double oneValue;
        while (outss >> oneValue) {
            targetVals.push_back(oneValue);
        }
    }
}
