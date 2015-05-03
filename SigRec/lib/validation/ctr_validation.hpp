/*
Copyright(c) 2015 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIGREC_CTR_VALIDATION_HPP
#define SIGREC_CTR_VALIDATION_HPP

#include "../model/ctr.h"
#include "metrics.hpp"
#include "cross_validation.hpp"
#include "SigDM/lib/ratings/sparse_boolean_matrix.hpp"
#include "SigUtil/lib/functional/filter.hpp"
#include "SigUtil/lib/tools/histgram.hpp"
//#include <boost/numeric/ublas/matrix_sparse.hpp>

namespace sigrec
{

template <>
class CrossValidation<CTR> : public CrossValidationBase<CTR::RatingValueType>
{
public:
	using Histgram = std::shared_ptr<sig::Histgram<double, 100>>;

	struct DetailInfo : public DetailInfoBase
	{
		uint const user_num_;
		uint const item_num_;
		const DocumentSetPtr docs_;
		const CTRHyperParamPtr hparam_;
		std::vector<Histgram> hist_;

	public:
		DetailInfo(uint user_num, uint item_num, DocumentSetPtr docs, CTRHyperParamPtr param, bool is_user_test, uint div_num)
			: DetailInfoBase(is_user_test, div_num), user_num_(user_num), item_num_(item_num), docs_(docs), hparam_(param){}
		DetailInfo(DetailInfo const&) = delete;

		void save(sig::FilepathString path) const override{
			DetailInfoBase::save(path);
			{
				std::ofstream ofs(path, std::ios::out | std::ios::app);
				ofs << std::endl;
				ofs << "number of users: " << user_num_ << std::endl;
				ofs << "number of items: " << item_num_ << std::endl;
				ofs << "number of topics: " << hparam_->topic_num_ << std::endl;
				ofs << "number of tokens: " << docs_->getTokenNum() << std::endl;
				ofs << "number of unique words: " << docs_->getWordNum() << std::endl;
				if ((!hparam_->theta_.empty()) && (!hparam_->beta_[0].empty())) ofs << "theta size: " << hparam_->theta_.size() << " * " << hparam_->theta_[0].size() << std::endl;
				if ((!hparam_->beta_.empty()) && (!hparam_->beta_[0].empty())) ofs << "phi(beta) size: " << hparam_->beta_.size() << " * " << hparam_->beta_[0].size() << std::endl;
				ofs << "positive weight in updating parameter U, V: " << hparam_->a_ << std::endl;
				ofs << "negative weight in updating parameter U, V: " << hparam_->b_ << std::endl;
				ofs << "gaussian variance which regularize user vector: " << hparam_->lambda_u_ << std::endl;
				ofs << "gaussian variance which regularize item vector: " << hparam_ ->lambda_v_ << std::endl;
				ofs << std::endl;
				ofs << "histgram of estimate ratings: " << std::endl;
			}
			for (auto const& e : hist_) e->print(path, false);
		}
	};

	using DetailInfoPtr = std::shared_ptr<DetailInfo>;

private:
	std::vector<CTRPtr> models_;
	CTRHyperParamPtr hparam_;
	DocumentSetPtr docs_;
	std::shared_ptr<DetailInfo> detail_info_sub_;
	
public:
	CrossValidation(
		uint div_num,
		bool is_user_test,
		CTRHyperParamPtr hparam,
		DocumentSetPtr docs,
		SparseRatingMatrixPtr<CTR::RatingValueType> ratings,
		uint max_iter,
		uint min_iter,
		FilepathString save_info_dir,
		bool is_save_param
	)
		: CrossValidationBase(std::make_shared<DetailInfo>(ratings->userSize(), ratings->itemSize(), docs, hparam, is_user_test, div_num)),
		models_(0), hparam_(hparam), docs_(docs), detail_info_sub_(std::dynamic_pointer_cast<DetailInfo>(detail_info_))
	{
		rating_chunks_ = devide_random(*ratings);
		//rating_chunks_ = devide_adjusted_random(*ratings);

		auto train_model = [=](uint i){
			uint const vsize = detail_info_sub_->is_user_test_ ? detail_info_sub_->user_num_ : detail_info_sub_->item_num_;
			auto model = CTR::makeInstance(hparam_, docs_, SparseBooleanMatrix::makeInstance(join_chunc(vsize, rating_chunks_, i)), i);
			model->train(max_iter, min_iter, save_info_dir, is_save_param);
			return model;
		};

		auto estimate_check = [](CTRPtr model){
			auto hist = std::make_shared<sig::Histgram<double, 100>>(-10, 10);

			for(uint u = 0, us = model->getUserNum(); u < us; ++u){
				for(uint i = 0, is = model->getItemNum(); i < is; ++i){
					double v = model->estimate(u, i);
					hist->count(v);
				}
			}
			return hist;
		};

		uint const n = detail_info_sub_->div_num_;
		const auto sqn =  sig::seqn(0u, 1u, n);

		models_ = run_parallel<CTRPtr>(train_model, n, sqn.begin());
		detail_info_sub_->hist_ = run_parallel<Histgram>(estimate_check, n, models_.begin());

		detail_info_sub_->save(save_info_dir + SIG_TO_FPSTR("validation info.txt"));
	}

	// F: R evaluation_func(CTRPtr model, std::vector<RatingContainer_>& test_set, bool is_user_test, double threshold)
	template <class F>
	auto run(F&& evaluation_func)
	{
		using R = decltype(sig::impl::eval(
			std::forward<F>(evaluation_func),
			std::declval<CTRPtr>(), 
			rating_chunks_[0],
			detail_info_sub_
		));
	
		auto validation = [&](uint test_index)
		{
			return std::forward<F>(evaluation_func)(models_[test_index], rating_chunks_[test_index], detail_info_sub_);
		};		
		
		std::vector<R> result;

		uint const n = detail_info_sub_->div_num_;
		return run_parallel<R>(validation, n, sig::seqn(0u, 1u, n).begin());
	}
};

template <class Derived>
struct CTR_PR_IMPL
{
	using RatingContainer_ = RatingContainer<CTR::RatingValueType>;

	CTR_PR_IMPL(sig::Maybe<uint> top_n, sig::Maybe<double> threshold) : top_n_(top_n), threshold_(threshold){}
	virtual ~CTR_PR_IMPL() = default;

	auto operator()(CTRPtr model, std::vector<RatingContainer_>& test_set, CrossValidation<CTR>::DetailInfoPtr info) const->double
	{
		const bool is_user_test = info->is_user_test_;
		auto get_id = [](CTR::RatingPtr_ const& rp, bool is_user) { return is_user ? rp->user_id_ : rp->item_id_; };

		std::vector<double> result;

		for (Id id = 0, size = test_set.size(); id < size; ++id){
			if(is_user_test ? model->getUserRatingNum(id) <= 1 : model->getItemRatingNum(id) <= 1) continue;

			auto est = model->recommend(id, is_user_test, top_n_, threshold_);
			if (est.size() < 1) continue;

			auto val = static_cast<const Derived*>(this)->impl(
				sig::map([](CTR::EstValueType const& e) { return e.first; }, est),
				sig::map([&](CTR::RatingPtr_ const& e) { return get_id(e, !is_user_test); }, /*test_set[id]),*/sig::filter([](RatingContainer_::value_type const& r) { return !r->is_duplicate_; }, test_set[id])),
				false,
				std::less<uint>()
			);
			if(val){
				result.push_back(*val);
			}
		}

		return sig::average(result);
	}

private:
	sig::Maybe<uint> top_n_;
	sig::Maybe<double> threshold_;
};

// precision for user (item recommendation)
template <>
struct Precision<CTR> : public PrecisionImpl, public CTR_PR_IMPL<Precision<CTR>>
{
	Precision(sig::Maybe<uint> top_n, sig::Maybe<double> threshold) : CTR_PR_IMPL(top_n, threshold){}
};

template <>
struct Recall<CTR> : public RecallImpl, public CTR_PR_IMPL<Recall<CTR>>
{
	Recall(sig::Maybe<uint> top_n, sig::Maybe<double> threshold) : CTR_PR_IMPL(top_n, threshold){}
};

template <>
struct F_Measure<CTR> : public F_MeasureImpl, private CTR_PR_IMPL<Precision<CTR>>, private CTR_PR_IMPL<Recall<CTR>>
{
	F_Measure(sig::Maybe<uint> top_n, sig::Maybe<double> threshold) : CTR_PR_IMPL<Precision<CTR>>(top_n, threshold), CTR_PR_IMPL<Recall<CTR>>(top_n, threshold){}

	double operator()(CTRPtr model, std::vector<RatingContainer<CTR::RatingValueType>>& test_set, CrossValidation<CTR>::DetailInfoPtr info) const
	{
		return impl(
			CTR_PR_IMPL<Precision<CTR>>::operator()(model, test_set, info),
			CTR_PR_IMPL<Recall<CTR>>::operator()(model, test_set, info)
		);
	}
};

template <>
struct AveragePrecision<CTR> : public AveragePrecisionImpl, public CTR_PR_IMPL<AveragePrecision<CTR>>
{
	AveragePrecision(sig::Maybe<uint> top_n, sig::Maybe<double> threshold) : CTR_PR_IMPL(top_n, threshold) {}
};

template <>
struct CatalogueCoverage<CTR> : public CatalogueCoverageImpl
{
	CatalogueCoverage(sig::Maybe<uint> top_n, sig::Maybe<double> threshold) : top_n_(top_n), threshold_(threshold) {}

	double operator()(CTRPtr model, std::vector<RatingContainer<CTR::RatingValueType>>& test_set, CrossValidation<CTR>::DetailInfoPtr info) const
	{
		const bool is_user_test = info->is_user_test_;
		double result = -1;
		std::vector<std::vector<Id>> ests;

		for (Id id = 0, size = test_set.size(); id < size; ++id) {
			auto est = model->recommend(id, is_user_test, top_n_, threshold_);
			if (est.size() < 1) continue;

			ests.push_back(sig::map([](CTR::EstValueType const& e) { return e.first; }, est));
		}

		auto val = this->impl(ests, is_user_test ? model->getItemNum() : model->getUserNum());
		if (val) result = *val;

		return result;
	}

private:
	sig::Maybe<uint> top_n_;
	sig::Maybe<double> threshold_;
};

template <>
struct InterUserDiversity<CTR> : public InterUserDiversityImpl
{
	InterUserDiversity(sig::Maybe<uint> top_n) : top_n_(top_n) {}

	double operator()(CTRPtr model, std::vector<RatingContainer<CTR::RatingValueType>>& test_set, CrossValidation<CTR>::DetailInfoPtr info) const
	{
		const bool is_user_test = info->is_user_test_;
		double result = -1;
		std::vector<std::vector<Id>> ests;

		for (Id id = 0, size = test_set.size(); id < size; ++id) {
			auto est = model->recommend(id, is_user_test, top_n_, Nothing(double{}));
			if (est.size() < 1) continue;

			auto tmp = sig::map([](CTR::EstValueType const& e) { return e.first; }, est);
			sig::sort(tmp, std::less<uint>());
			ests.push_back(tmp);
		}

		auto val = this->impl(ests, top_n_ ? *top_n_ : ests.begin()->size(), true, std::less<uint>());
		if (val) result = *val;

		return result;
	}

private:
	sig::Maybe<uint> top_n_;
};

template <>
struct ListPersonalizationMetric<CTR> : public ListPersonalizationMetricImpl
{
	ListPersonalizationMetric(sig::Maybe<uint> top_n, sig::Maybe<double> threshold) : top_n_(top_n), threshold_(threshold) {}

	double operator()(CTRPtr model, std::vector<RatingContainer<CTR::RatingValueType>>& test_set, CrossValidation<CTR>::DetailInfoPtr info) const
	{
		const bool is_user_test = info->is_user_test_;
		double result = -1;
		std::vector<std::vector<Id>> ests;

		for (Id id = 0, size = test_set.size(); id < size; ++id) {
			auto est = model->recommend(id, is_user_test, top_n_, threshold_);
			if (est.size() < 1) continue;

			ests.push_back(sig::map([](CTR::EstValueType const& e) { return e.first; }, est));
		}

		auto val = this->impl(ests);
		if (val) result = *val;

		return result;
	}

private:
	sig::Maybe<uint> top_n_;
	sig::Maybe<double> threshold_;
};

}
#endif
