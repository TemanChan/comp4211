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
		weights[i] = (rand() % 1000) / 1000.0; // weights range [0, 1]
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
		currentOutput += (weights[i] * inputs[i].getOutput());
	}
	//cout << "output: " << currentOutput;
	currentOutput = 1.0 / (1 + exp(-1 * currentOutput));
	//cout << "\t sigmoid(output): " << currentOutput << endl;
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
	for(int i=0; i<inputs.size(); ++i){
		double delta = eta * sigma * inputs[i].getOutput();
		weights[i] += delta;
		//cout << "delta: " << delta << endl;
	}
}

double Neuron::getWeight(int idx) const
{
	return weights[idx];
}
