/*
Copyright(c) 2015 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIGREC_MATRIX_FACTORIZATION_INTERFACE_HPP
#define SIGREC_MATRIX_FACTORIZATION_INTERFACE_HPP

#include "../sigrec.hpp"
#include "SigDM/lib/ratings/rating.hpp"

namespace sigrec
{

class MatrixFactorization
{
protected:
	MatrixFactorization() = default;
	MatrixFactorization(MatrixFactorization&&) = delete;	// コピー操作とムーブ操作を制限(スマートポインタで扱うように限定)

public:
	virtual ~MatrixFactorization() = default;

	virtual void train(uint num_iteration) = 0;

	virtual void train(uint num_iteration, std::function<void(MatrixFactorization const&)> callback) = 0;

	virtual double estimate(uint user_id, uint item_id) const = 0;

	virtual double absolute_error() const = 0;

	//virtual double relative error() const = 0;

};

}
#endif