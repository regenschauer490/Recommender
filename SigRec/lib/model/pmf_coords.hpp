/*
Copyright(c) 2015 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIGREC_PMF_COORDS_HPP
#define SIGREC_PMF_COORDS_HPP

#include "../sigrec.hpp"
#include "SigDM/lib/ratings/rating.hpp"
#include "SigUtil/lib/tools/random.hpp"

namespace sigrec
{

/**
\brief
	Probabilistic Matrix Factorization model

\details
	
	[1] Hu, Y., Koren, Y. and Volinsky, C.: Collaborative Filtering for Implicit Feedback Datasets,	Proc. IEEE ICDM (2008)
*/
template <class ValueType>
class PMF_Coords : public MatrixFactorization
{
	using RatingMatrix_ = SparseRatingMatrixPtr<ValueType>;
	using Matrix_ = BlasMatrix<double>;

private:
	Ratings_ const& ratings_;	// U * V

	uint const U_;	// number of users
	uint const V_;	// number of items
	uint const K_;	// number of latent factors

	double const alpha_;	// learning rate of SGD
	double const lambda_;	// penalty parameter of objective function

	Matrix_ mat_u_;	// U * K
	Matrix_ mat_v_;	// V * K

	double error_;
	sig::SimpleRandom<double> random_;

private:
	void init(std::function<void(Matrix_&)> const& init_mat_func){
		init_mat_func(mat_u_);
		init_mat_func(mat_v_);
	}

	virtual void update() = 0;

public:
	template <class F>
	MatrixFactorization(Ratings const& ratings, F const& init_mat_func, uint num_factor, double alpha = 0.001, double lambda = 0.001)
	:	U_(ratings.size()), V_(ratings[0].size()), K_(num_factor), alpha_(alpha), lambda_(lambda), ratings_(ratings),
		mat_u_(init_mat_func(U_, K_)), mat_v_(init_mat_func(V_, K_)), random_(0, 1, MF_DEBUG_MODE)
	{
		init(init_mat_func);
	}

	template <class F1, class F2>
	void train(uint iteration, F1 const& error_func, F2 const& update_func){
		for (uint i = 0; i < iteration; ++i){
			update(error_func, update_func);
		}
	}

	template <class F>
	double estimate(uint u, uint v, F const& inner_prod) const{
		return inner_prod(mat_u_[u], mat_v_[v]);
	}

	double error() const {
		return error_;
	}

	template <class S>
	void print_factor_u(S& stream) const {
		for (uint u = 0; u < U_; ++u) {
			for (uint k = 0; k < K_; ++k) stream << mat_u_[u][k] << " ";
			stream << std::endl;
		}
	}
	template <class S>
	void print_factor_v(S& stream) const {
		for (uint v = 0; v < V_; ++v){
			for (uint k = 0; k < K_; ++k) stream << mat_v_[v][k] << " ";
			stream << std::endl;
		}
	}
};

}
#endif