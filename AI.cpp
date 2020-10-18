#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>

using namespace std;

struct Connection {
  double weight;
  double deltaWeight;
};

class Neuron;
typedef vector<Neuron> Layer;

class Neuron {
  public:
    Neuron(unsigned numOutput, unsigned myIndex);
    void setOutputVal(double val) {n_outputVal = val;}
    double getOutputVal(void) const {return n_outputVal;}
    void feedForward(const Layer &prevLayer);
  private:
    static double transferFunction(double x);
    static double transferFunctionDerivative(double x);
    static double randomWeight(void) {return rand() / double(RAND_MAX);}
    double n_outputVal;
    vector<Connection> n_outputWeights;
    unsigned n_myIndex;
};

double Neuron::transferFunction(double x) {
  return tanh(x);
}

double Neuron::transferFunctionDerivative(double x) {
  return 1.0 - x * x;
}

Neuron::Neuron (unsigned numOutputs, unsigned myIndex) {
  for (unsigned c = 0; c < numOutputs; ++c) {
    n_outputWeights.push_back(Connection());
    n_outputWeights.back().weight = randomWeight();
  };

  n_myIndex = myIndex;

}

void Neuron::feedForward(const Layer &prevLayer) {
  double sum = 0.0;

  for (unsigned n = 0; n < prevLayer.size(); ++n) {
    sum += prevLayer[n].getOutputVal() *
      prevLayer[n].n_outputWeights[n_myIndex].weight;
  }

  n_outputVal = Neuron::transferFunction(sum);
}

class Net {
  public:
    Net(const vector<unsigned> &topology);
    void feedForward(const vector<double> &inputVals);
    void backProp(const vector<double> &targetVals) {};
    void getResults(vector<double> &resultVals) const {};

  private:
    vector<Layer> n_layers;
};

void Net::feedForward(const vector<double> &inputVals) {
  assert(inputVals.size() == n_layers[0].size() - 1);

  for (unsigned i = 0; i < inputVals.size(); ++i) {
    n_layers[0][i].setOutputVal(inputVals[i]); 
  }

  for (unsigned layerNum = 1; layerNum < n_layers.size(); ++layerNum) {
    Layer &prevLayer = n_layers[layerNum - 1];
    for (unsigned n = 0; n < n_layers[layerNum].size() - 1; ++n) {
      n_layers[layerNum][n].feedForward(prevLayer);
    }
  }
};

Net::Net(const vector<unsigned> &topology) {
  unsigned numLayers = topology.size();
  for(unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
    unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];
    n_layers.push_back(Layer());
    for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
      n_layers.back().push_back(Neuron(numOutputs, neuronNum)); 
      cout << "Made a neuron!" << endl;
    }

  }
}

int main() {

  vector<unsigned> topology;
  topology.push_back(3);
  topology.push_back(2);
  topology.push_back(1);
  Net myNet(topology);

  vector<double> inputVals;
  myNet.feedForward(inputVals);

  vector<double> targetVals;
  myNet.backProp(targetVals);

  vector<double> resultVals;
  myNet.getResults(resultVals);
}