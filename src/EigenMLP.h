#pragma once


class NN {
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
		NN();
	Eigen::Vector <double, 784> a0;
	Eigen::Vector <double, 10> a1;
	Eigen::Vector <double, 10> a2;

	Eigen::Matrix <double, 10, 784> W1;
	Eigen::Matrix <double, 10, 10> W2;

	Eigen::Vector <double, 10> b1;
	Eigen::Vector <double, 10> b2;

	Eigen::Vector <double, 10> Z1;
	Eigen::Vector <double, 10> Z2;


	Eigen::Vector <double, 10> a;

	Eigen::Vector<double, 10> softmax(Eigen::Vector<double, 10> Z);
	Eigen::Vector<double, 10> ReLu(Eigen::Vector<double, 10> Z);

	Eigen::Vector<double, 10> one_hot(int y);



	void forward_prop();
	void init_params();

};
