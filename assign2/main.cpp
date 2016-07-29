#include <cstring>
#include <cstdlib>
#include <cmath>
#include <queue>
#include "TrainingData.h"
#include "Net.h"
using namespace std;

void showVectorVals(string label, vector<double> &v)
{
    cout << label << " ";
    for (unsigned i = 0; i < v.size(); ++i) {
        cout << v[i] << " ";
    }

    cout << endl;
}

int main(int argc, char **argv)
{
    double ETA = 0.5; // learning rate
    double THETA = 0.01;
    string FILENAME("data.txt");
    string actFunc("s");

    queue<double> lastTenErrors;
    double currentError;
    double errorSum = 0;

    assert(argc % 2); // "The number of arguments is incorrect"
    for(int i=1; i<argc; ++i){
        if(strcmp(argv[i], "-e") == 0){
            ETA = atof(argv[++i]);
        }else if(strcmp(argv[i], "-f") == 0){
            FILENAME = string(argv[++i]);
        }else if(strcmp(argv[i], "-af") == 0){
            actFunc = string(argv[++i]);
        }else if(strcmp(argv[i], "-t") == 0){
            THETA = atof(argv[++i]);
        }
    }
    TrainingData trainData(FILENAME);

    // e.g., { 2, 4, 1 }
    vector<unsigned> topology;

    // get the structure of the net
    trainData.getTopology(topology);

    Net myNet(topology, ETA, actFunc);

    vector<double> inputVals, targetVals, resultVals;
    int trainingIteration = 0;

    while(trainingIteration < 100){
        ++trainingIteration;
        cout << endl << "Iteration " << trainingIteration << endl;

        double maxError = 0;
        while(!trainData.isEof()){
            trainData.getNextSample(inputVals, targetVals);

            assert(inputVals.size() == topology[0]);
            showVectorVals("Inputs:", inputVals);

            // feed Forward	
            myNet.feedForward(inputVals);

            // Collect the net's actual output results:
            myNet.getResults(resultVals);
            showVectorVals("Outputs:", resultVals);

            // Train the net what the outputs should have been:
            showVectorVals("Targets:", targetVals);
            assert(targetVals.size() == topology.back());

            myNet.backProp(targetVals);

            double error = myNet.getError();
            cout << "Error: " << error << endl;

            if(error > maxError) maxError = error;
        }
        if(maxError < THETA) break;
        else trainData.restart();
    }

    // test performance
    cout << endl << "Total # of iterations: " << trainingIteration << endl;
    vector<vector<double>> inputs = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
    vector<double> targets = { 0, 1, 1, 0 };
    vector<double> outputs;
    bool allCorrect = true;
    for(int i=0; i<4; ++i){
        myNet.feedForward(inputs[i]);
        showVectorVals("Inputs:", inputs[i]);
        myNet.getResults(outputs);
        showVectorVals("Outputs:", outputs);
        if(abs(outputs[0] - targets[i]) > 0.1){
            allCorrect = false;
            //break;
        }
    }
    cerr << (allCorrect ? "True" : "False") << endl;
    cout << endl << "Training Complete" << endl;
}
