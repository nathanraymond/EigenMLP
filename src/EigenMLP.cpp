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

	a1 = ReLu(Z1);

	Z2 = W2 * a1 + b2;

	a2 = softmax(Z2);

	a = a2;

}

Eigen::Vector <double, 10> NN::softmax(Eigen::Vector <double, 10> Z) {

	Z = Z.array().exp() / Z.array().exp().sum();

	return Z;
}

Eigen::Vector<double, 10> NN::ReLu(Eigen::Vector<double, 10> Z) {

	for (int i = 0; i < Z.size(); i++) {

		Z(i) = std::max(0.0, Z(i));

	}
	return Z;
}

Eigen::Vector<double, 10> NN::one_hot(int y){

	Eigen::Vector <double, 10> one_hot_Y = Eigen::VectorXd::Zero(10);

	one_hot_Y(y) = 1;

	return one_hot_Y;
}





int main()
{

	NN* n = new NN;

	n->init_params();

	n->forward_prop();


	std::cout << "\n";
	std::cout << (n->a) << "\n";
	std::cout << "\n";
	Eigen::Vector <double, 10> one_hot_Y = Eigen::VectorXd::Zero(10);

	one_hot_Y(0) = 1;

	std::cout << one_hot_Y << "\n";
}