#include <cstdlib>
#include <iostream>
#include <ctime>
#include <cmath>
#include "Neuron.h"
using namespace std;

Neuron::Neuron(int numOfInputs, const string &actFunc)
{
	srand(time(NULL));
	weights = vector<double>(numOfInputs);
	for(int i=0; i < weights.size(); ++i){
		weights[i] = (rand() % 1000) / 1000.0; // weights range [0, 1]
	}

	if(actFunc.compare("r") == 0){
		activationFunction = ReLU;
	}else if(actFunc.compare("lr") == 0){
		activationFunction = leakyReLU;
	}else{
		activationFunction = sigmoid;
	}
	sigma = 0;
	currentOutput = 0;
}

double Neuron::sigmoid(double x)
{
	return 1.0 / (1 + exp(-1 * x));
}

double Neuron::ReLU(double x)
{
	return x < 0 ? 0 : x;
}

double Neuron::leakyReLU(double x)
{
    return x < 0 ? 0.001*x : x;
}

void Neuron::setOutput(double output)
{
	currentOutput = output;
}

double Neuron::computeOutput(std::vector<Neuron> inputs)
{
	currentOutput = 0;
	for(int i=0; i<inputs.size(); ++i){
		currentOutput += (weights[i] * inputs[i].getOutput());
	}
	//cout << "output: " << currentOutput;
	currentOutput = activationFunction(currentOutput);
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
