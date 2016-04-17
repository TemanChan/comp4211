#include "Net.h"
using namespace std;

Net::Net(const vector<unsigned> &topology, const double myEta)
{
	eta = myEta;
	//cout << "In constructor, eta = " << eta << endl;
	layers = vector<vector<Neuron> >(topology.size());
	int numInputs = 0;
	for(int i=0; i<topology.size(); ++i){
		layers[i] = vector<Neuron>(topology[i], Neuron(numInputs));
		numInputs = topology[i];
	}
}

void Net::feedForward(const vector<double> &inputVals)
{
	for(int i=0; i<inputVals.size(); ++i){
		layers[0][i].setOutput(inputVals[i]);
	}
	for(int i=1; i<layers.size(); ++i){
		for(int j=0; j<layers[i].size(); ++j){
			layers[i][j].computeOutput(layers[i-1]);
		}
	}
}

void Net::backProp(const vector<double> &targetVals)
{
	currentTargets = targetVals;
	int n = layers.size();
	for(int i=0; i<layers[n-1].size(); ++i){
		Neuron &neuron = layers[n-1][i];
		double output = neuron.getOutput();
		double sigma = output * (1 - output) * (targetVals[i] - output);
		neuron.setSigma(sigma);
	}

	for(int i=n-2; i>=1; --i){
		for(int j=0; j<layers[i].size(); ++j){
			Neuron &neuron = layers[i][j];
			double output = neuron.getOutput();
			double sum = 0;
			for(int k=0; k<layers[i+1].size(); ++k){
				sum += layers[i+1][k].getWeight(j) * layers[i+1][k].getSigma();
			}
			double sigma = output * (1 - output) * sum;
			neuron.setSigma(sigma);
		}
	}

	// update weights
	for(int i=1; i<layers.size(); ++i){
		for(int j=0; j<layers[i].size(); ++j){
			Neuron &neuron = layers[i][j];
			//cout << "In backProp, eta = " << eta << endl;
			neuron.updateWeights(layers[i-1], eta);
		}
	}
}

void Net::getResults(vector<double> &resultVals) const
{
	int numOutputs = layers[layers.size()-1].size();
	resultVals = vector<double>(numOutputs);
	for(int i=0; i<numOutputs; ++i){
		resultVals[i] = layers[layers.size()-1][i].getOutput();
	}
}

double Net::getError(void) const
{
	double err = 0;
	for(int i=0; i<currentTargets.size(); ++i){
		double output = layers[layers.size()-1][i].getOutput();
		err += (currentTargets[i] - output) * (currentTargets[i] - output);
	}
	return err;
}
