#ifndef NEURON_H
#define NEURON_H

#include <vector>

class Neuron
{
public:
	Neuron(int numOfInputs, const std::string &actFunc = "s");
	void setOutput(double output);
	double computeOutput(std::vector<Neuron> inputs);
	double getOutput() const;
	void updateWeights(std::vector<Neuron> inputs, double eta);
	double getWeight(int idx) const;
	void setSigma(double newSigma);
	double getSigma() const;

private:
	static double sigmoid(double x);
	static double ReLU(double x);
	static double leakyReLU(double x);
	double (*activationFunction)(double);
	std::vector<double> weights;
	double currentOutput;
	double sigma;
};


#endif
