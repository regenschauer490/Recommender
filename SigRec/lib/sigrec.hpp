/*
Copyright(c) 2015 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIGREC_HPP
#define SIGREC_HPP

#include "SigUtil/lib/sigutil.hpp"
#include "SigUtil/lib/helper/maybe.hpp"

#if (!SIG_USE_BOOST)
static_assert(false, "This library requires Boost C++ Libraries");
#endif


namespace sigrec
{

#define SIG_USE_SIGTM 1			// TopicModelライブラリ(https://github.com/regenschauer490/TopicModel)を使用するか (CTRで必要)
#define SIG_USE_EIGEN 1			// 行列演算にライブラリのEigenを使用するか（処理速度向上）

const bool FixedRandom = true;		// 乱数を固定するか(テスト用)
}

#if SIG_USE_EIGEN
#include "Eigen/Core"
#else
#include "SigUtil/lib/calculation/ublas.hpp"
#endif

namespace sigrec
{
using sig::uint;
using sig::FilepassString;

using sig::Maybe;
using sig::Just;
using sig::Nothing;
using sig::isJust;
using sig::fromJust;


using ItemId = uint;
using UserId = uint;

#if SIG_USE_EIGEN
template<class T> using Vector = typename std::conditional<
	std::numeric_limits<T>::is_integer,
	Eigen::VectorXi,
	Eigen::VectorXd
>::type;

template<class T> using Matrix = typename std::conditional<
	std::numeric_limits<T>::is_integer,
	Eigen::MatrixXi,
	Eigen::MatrixXd
>::type;
		
#else

template<class T> using Vector = vector_u<T>;	
template<class T> using Matrix = matrix_u<T>;
#endif


template<class T> using VectorI = Vector<T>;	// item	
template<class T> using VectorU = Vector<T>;	// user

}
#endif