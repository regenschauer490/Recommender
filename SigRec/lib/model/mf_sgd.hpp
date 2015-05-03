/*
Copyright(c) 2015 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIGREC_MATRIX_FACTORIZATION_HPP
#define SIGREC_MATRIX_FACTORIZATION_HPP

#include "../sigrec.hpp"
#include "SigDM/lib/ratings/sparse_matrix.hpp"
#include "SigDM/lib/util/eigen_ublas_util.hpp"
#include "SigUtil/lib/tools/random.hpp"

namespace sigrec
{
using sigdm::RatingPtr;
using sigdm::SparseRatingMatrix;
using sigdm::SparseRatingMatrixPtr;
using sigdm::SparseBooleanMatrix;
using sigdm::SparseBooleanMatrixPtr;

/**
\brief
	Matrix Factorization model trained by Stochastic Gradient Descent

\details
	This model recommends each user to ranked items by collaborative filtering.
	To estimate missing ratings, learn model-parameters from known ratings by following optimazation method.
	This model minimizes the regularized squared error on the set of known ratings.
	To achieve this, use SGD(stochastic gradient descent) to update parameters iteratively until convergence.

	[1] Koren, Y., Bell, R. and Volinsky, C.: Matrix factorization techniques for recommender systems, Computer (2009)
*/
template <class ValueType>
class MF_SGD : public MatrixFactorization
{
	using RatingMatrix_ = SparseRatingMatrixPtr<ValueType>;
	using Matrix_ = BlasMatrix<double>;

private:
	RatingMatrix_ const& ratings_;	// U * V

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
	void init(){
		for (uint k = 0; k < K_; ++k) {
			for (uint u = 0; u < U_; ++u) mat_u_[u][k] = random_();
			for (uint v = 0; v < V_; ++v) mat_v_[v][k] = random_();
		}
	}

	void update(){
		double soe = 0;

		for (uint u = 0; u < U_; ++u){
			for (uint v = 0; v < V_; ++v){
				if (ratings_[u][v] == 0) continue;

				double const error = sigdm::impl::at_(*ratings_, u, v) - sigdm::impl::inner_prod(mat_u_[u], mat_v_[v]);
				soe += error;

				lf1 += a * (e * lf2 - l * lf1)
				mat_u_[u] += alpha_ * (error * mat_v_[v] - lambda_ *mat_u_[u]);
				mat_v_[v] += alpha_ * (error * mat_u_[u] - lambda_ * mat_v_[v]);
			}
		}
		if(MF_DEBUG_MODE) std::cout << soe << std::endl;
		
		//bool local_conv = soe < error_;
		error_ = soe;
	}

private:
	MF_SGD(RatingMatrix_ const& ratings, uint num_factor, Maybe<double> alpha, Maybe<double> lambda)
		: ratings_(ratings), U_(ratings.size()), V_(ratings[0].size()), K_(num_factor),
		  alpha_(isJust(alpha) ? fromJust(alpha) : default_mf_alpha), lambda_(isJust(lambda) ? fromJust(lambda) : default_mf_lambda),
		  random_(0, 1, MF_DEBUG_MODE)
	{
		init();
	}

public:
	/**
	\brief
		@~japanese ファクトリ関数	\n
		@~english factory function	\n

	\details
		@~japanese

		@~english
	*/
	static auto makeInstance(
		SparseRatingMatrixPtr<ValueType> const& ratings,
		uint num_factor,
		Maybe<double> alpha = Nothing<double>(),
		Maybe<double> lambda = Nothing<double>()
	) ->std::shared_ptr<MF_SGD>
	{
		return std::shared_ptr<MF_SGD>(new MF_SGD(ratings, num_factor, alpha, lambda));
	}
	
	void train(uint iteration) override{
		for (uint i = 0; i < iteration; ++i){
			update();
		}
	}


	double estimate(uint user_id, uint item_id) const override{
		return sigdm::impl::inner_prod(mat_u_[user_id], mat_v_[item_id]);
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