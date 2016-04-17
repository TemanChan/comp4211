#include <cstdlib>
#include <iostream>
#include <ctime>
#include <cmath>
#include "Neuron.h"
#include "errMsg.h"
using namespace std;

Neuron::Neuron(int numOfInputs)
{
	srand(time(NULL));
	weights = vector<double>(numOfInputs);
	for(int i=0; i < weights.size(); ++i){
		weights[i] = (rand() % 2000 - 1000) / 100000.0;
	}
	sigma = 0;
	currentOutput = 0;
}

void Neuron::setOutput(double output)
{
	currentOutput = output;
}

double Neuron::computeOutput(std::vector<Neuron> inputs)
{
	if(inputs.size() != weights.size()){
		errMsg("dimension does not match");
	}
	currentOutput = 0;
	for(int i=0; i<inputs.size(); ++i){
		currentOutput += weights[i] * inputs[i].getOutput();
	}
	currentOutput = 1.0 / (1 + exp(-1 * currentOutput));
	return currentOutput;
}

double Neuron::getOutput() const
{
	return currentOutput;
}

void Neuron::setSigma(double newSigma)
{
	sigma = newSigma;
}

double Neuron::getSigma() const
{
	return sigma;
}

void Neuron::updateWeights(std::vector<Neuron> inputs, double eta)
{
	if(inputs.size() != weights.size()){
		errMsg("dimension does not match");
	}
	//cout << "In Neuron updateWeights, eta = " << eta << endl;
	for(int i=0; i<inputs.size(); ++i){
		//cout << "eta: " << eta << ", sigma: " << sigma << ", output: " << inputs[i].getOutput() << endl;
		weights[i] += eta * sigma * inputs[i].getOutput();
		cout << weights[i] << " ";
	}
	//cout << endl;
}

double Neuron::getWeight(int idx) const
{
	return weights[idx];
}
