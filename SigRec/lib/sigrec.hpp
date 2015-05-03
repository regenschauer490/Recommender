/*
Copyright(c) 2015 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIGREC_HPP
#define SIGREC_HPP

/*--------------------------------------- User Option --------------------------------------------------------------------*/

#define SIG_USE_SIGTM	1	// TopicModelライブラリ(https://github.com/regenschauer490/TopicModel)を使用するか (CTRで必要)

/*------------------------------------------------------------------------------------------------------------------------*/


#include "SigDM/lib/sigdm.hpp"
#include "SigUtil/lib/sigutil.hpp"
#include "SigUtil/lib/helper/maybe.hpp"

namespace sigrec
{
bool const FixedRandom = true;	// 乱数を固定するか(テスト用)

using sig::uint;
using sig::FilepathString;

using sig::Maybe;
using sig::Just;
using sig::Nothing;
using sig::isJust;
using sig::fromJust;

using sigdm::Id;
using sigdm::UserId;
using sigdm::ItemId;
using sigdm::VectorU;
using sigdm::VectorI;
using sigdm::VectorK;

using sigdm::BlasVector;
using sigdm::BlasMatrix;
using sigdm::BlasVectorU;
using sigdm::BlasVectorI;

double const default_mf_alpha = 0.001;
double const default_mf_lambda = 0.001;
double const log_lower_limit = -1000000;
}
#endif