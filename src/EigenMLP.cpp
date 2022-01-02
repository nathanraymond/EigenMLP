#include <Eigen/Dense>
#include <iostream>
#include "EigenMLP.h"

NN::NN() {

}

void NN::init_params() {
	a0.setRandom();

	W1.setRandom();
	W2.setRandom();

	b1.setOnes();
	b2.setOnes();
}

void NN::forward_prop() {
	Z1 = W1 * a0 + b1;
	Get_ReLu(Z1);
	a1 << ReLu;
	Z2 = W2 * a1 + b2;
	Z = Z2;
	Get_softmax(Z2);
	a2 << softmax;
	a = a2;

}

void NN::Get_ReLu(Eigen::Vector <double, 10> Z) {

	for (int i = 0; i < Z.size(); i++) {

		ReLu(i) = std::max(0.0, Z(i));

	}

}

void NN::Get_softmax(Eigen::Vector <double, 10> Z) {

	for (int i = 0; i < Z.size(); i++) {

		softmax(i) =  exp(Z(i)) / Z.array().exp().sum();

	}

};



int main()
{

	NN* n = new NN;

	n->init_params();

	n->forward_prop();


	std::cout << "\n";
	std::cout << n->a << "\n";



}