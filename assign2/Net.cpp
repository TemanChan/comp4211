#include "Net.h"
using namespace std;

Net::Net(const vector<unsigned> &topology, const double myEta, const string &actFunc)
{
	eta = myEta;
	layers = vector<vector<Neuron> >(topology.size());
	int numInputs = 0;
	for(int i=0; i<topology.size(); ++i){
		layers[i] = vector<Neuron>(topology[i], Neuron(numInputs, actFunc));
		numInputs = topology[i];
	}

	if(actFunc.compare("r") == 0){
		deriv = ReLUDeriv;
	}else if(actFunc.compare("lr") == 0){
		deriv = leakyReLUDeriv;
	}else{
		deriv = sigmoidDeriv;
	}
}


double Net::sigmoidDeriv(double y)
{
	return y * (1 - y);
}

double Net::ReLUDeriv(double y)
{
	return y > 0 ? 1 : 0;
}

double Net::leakyReLUDeriv(double y)
{
	return y > 0 ? 1 : 0.001;
}

void Net::feedForward(const vector<double> &inputVals)
{
	// the first layer is teh input layer
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
	// compute the sigmas of output layer
	for(int i=0; i<layers[n-1].size(); ++i){
		Neuron &neuron = layers[n-1][i];
		double output = neuron.getOutput();
		double sigma = deriv(output) * (targetVals[i] - output);
		//cout << "sigma: " << sigma << endl;
		neuron.setSigma(sigma);
	}

	// compute the sigmas of hidden layers
	for(int i=n-2; i>=1; --i){
		for(int j=0; j<layers[i].size(); ++j){
			Neuron &neuron = layers[i][j];
			double output = neuron.getOutput();
			double d = deriv(output);
			if(d == 0){
				neuron.setSigma(0);
			}else{
				double sum = 0;
				for(int k=0; k<layers[i+1].size(); ++k){
					sum += (layers[i+1][k].getWeight(j) * layers[i+1][k].getSigma());
				}
				double sigma = d * sum;
				//cout << "sigma: " << sigma << endl;
				neuron.setSigma(sigma);
			}
		}
	}

	// update weights
	for(int i=1; i<layers.size(); ++i){
		for(int j=0; j<layers[i].size(); ++j){
			Neuron &neuron = layers[i][j];
			neuron.updateWeights(layers[i-1], eta);
		}
	}
}

void Net::getResults(vector<double> &resultVals) const
{
	const vector<Neuron> &outputLayer = layers.back();
	int numOfOutputs = outputLayer.size();
	resultVals = vector<double>(numOfOutputs);
	for(int i=0; i<numOfOutputs; ++i){
		resultVals[i] = outputLayer[i].getOutput();
	}
}

double Net::getError(void) const
{
	double err = 0;
	const vector<Neuron> &outputLayer = layers.back();
	for(int i=0; i<currentTargets.size(); ++i){
		double output = outputLayer[i].getOutput();
		err += ((currentTargets[i] - output) * (currentTargets[i] - output));
	}
	return err;
}
