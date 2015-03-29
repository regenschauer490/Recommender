/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIGTM_CROSS_VALIDATION_H
#define SIGTM_CROSS_VALIDATION_H

#include "rating_matrix.hpp"
#include "SigUtil/lib/modify/shuffle.hpp"
//#include <boost/range/adapter/slice.hpp>

namespace sigtm
{

class CTR;

template <class T>
using RatingChunk = std::vector<std::vector<RatingContainer<T>>>;

template <class MODEL>
class CrossValidation;

// validation for rating matrix
template <class RatingType>
class CrossValidationBase
{
public:
	struct DetailInfoBase
	{
		const bool is_user_test_;
		const uint div_num_;
		int less_than_div_num_;
		double sparsity_;
		std::vector<double> chunk_sparsity_;

	public:
		DetailInfoBase(bool is_user_test, uint div_num) : is_user_test_(is_user_test), div_num_(div_num), less_than_div_num_(-1), sparsity_(-1){}
		DetailInfoBase(DetailInfoBase const&) = delete;
		virtual ~DetailInfoBase() = default;

		virtual void save(sig::FilepassString pass) const {
			std::ofstream ofs(pass);
			ofs << "this validation is for user test (1:true, 0:false): " << is_user_test_ << std::endl;
			ofs << "division number of dataset (number of chunks):  " << div_num_ << std::endl;
			ofs << "items less than division number in dataset: " << less_than_div_num_ << std::endl;
			ofs << "original rating matrix sparsity: " << sparsity_ << std::endl;
			ofs << "chunk matrix sparsity: " << std::endl;
			for (auto const& c : chunk_sparsity_) ofs << "\t" << c << std::endl;
		}
	};

	using DetailInfoPtr = std::shared_ptr<DetailInfoBase>;

protected:
	RatingChunk<RatingType> rating_chunks_;
	DetailInfoPtr detail_info_;

protected:
	CrossValidationBase(DetailInfoPtr detail_info) : detail_info_(detail_info){
		if (detail_info->div_num_ < 2) std::terminate();
	};
	virtual ~CrossValidationBase() = default;

	template <class T, class R = typename RatingChunk<T>::value_type>
	auto join_chunc(uint vec_size, RatingChunk<T> const& chunks, uint removed_index) ->R
	{
		R result(vec_size);

		for (uint c = 0; c < chunks.size(); ++c){
			if (c != removed_index){
				for(uint j = 0; j < vec_size; ++j){
					for (auto&& e : boost::make_iterator_range(std::begin(chunks[c][j]), std::end(chunks[c][j]))) result[j].push_back(e);

					//jr[i] = boost::join(boost::any_cast<const_joined_range>(jr[j]), std::begin(dataset[i][j]), std::end(dataset[i][j]));
				}
			}
		}

		return result;
		//return sig::map([](boost::any const& e){ return boost::any_cast<const_joined_range>(e); }, jr);
	}

	template <class T>
	auto devide_random(SparseRatingMatrixBase_<T> const& src) ->RatingChunk<T>
	{
		const uint n = detail_info_->div_num_;
		const bool is_user_test = detail_info_->is_user_test_;
		uint all_ct = 0;

		auto&& src_vecs = is_user_test ? src.getUsers() : src.getItems();
		std::vector<RatingPtr<T>> ratings;

		for (auto&& sv : src_vecs) {
			for (auto&& e : sv) {
				ratings.push_back(e);
				++all_ct;
			}
		}
		sig::shuffle(ratings);

		RatingChunk<T> chunks(n, std::vector<RatingContainer<T>>(is_user_test ? src.userSize() : src.itemSize()));
		const uint delta = ratings.size() / n;

		for (uint i = 0; i < n; ++i) {
			for (uint j = i * delta, ed = (i + 1) * delta, size = i != n - 1 ? ed : ratings.size(); j < size; ++j) {
				chunks[i][is_user_test ? ratings[j]->user_id_ : ratings[j]->item_id_].push_back(ratings[j]);
			}
		}

		const double matrix_size = src.userSize() * src.itemSize();
		detail_info_->sparsity_ = all_ct / matrix_size;
		for (auto const& c : chunks) detail_info_->chunk_sparsity_.push_back(delta / matrix_size);

		return std::move(chunks);
	}

	template <class T>
	auto devide_adjusted_random(SparseRatingMatrixBase_<T> const& src) ->RatingChunk<T>
	{
		const uint n = detail_info_->div_num_;
		const bool is_user_test = detail_info_->is_user_test_;
		uint ct = 0, all_ct = 0;

		auto&& src_vecs = is_user_test ? src.getUsers() : src.getItems();
		std::unordered_map<Id, std::vector<RatingPtr<T>>> ratings;

		auto get_id = [](RatingPtr<T> const& rp, bool is_user) { return is_user ? rp->user_id_ : rp->item_id_; };

		for (auto&& sv : src_vecs) {
			for (auto&& r : sv) {
				auto id = get_id(r, !is_user_test);
				if (ratings.count(id)) ratings[id].push_back(r);
				else ratings.emplace(id, std::vector<RatingPtr<T>>{r});
				++all_ct;
			}
		}

		RatingChunk<T> chunks(n, std::vector<RatingContainer<T>>(is_user_test ? src.userSize() : src.itemSize()));
		sig::SimpleRandom<uint> random(0, n - 1, FixedRandom);

		// for each items or users
		for (auto& e : ratings) {
			std::vector<RatingPtr<T>> rs = e.second;

			sig::shuffle(rs);

			if (rs.size() < n) {
				sig::SimpleRandom<uint> random2(0, rs.size() - 1, FixedRandom);

				for (uint i = 0; i < n; ++i) {
					auto& r = rs[random2()];
					chunks[i][get_id(r, is_user_test)].push_back(r);
					r->is_duplicate_ = true;
					/*for (uint j = 0; j < rs.size(); ++j) {
					chunks[i][rs[j]->user_id_].push_back(rs[j]);
					rs[j]->is_duplicate_ = true;
					}*/
				}
				++ct;
			}
			else {
				uint loop = rs.size() / n;
				uint i = 0;
				for (uint le = n * loop; i < le; ++i) {
					chunks[i%n][get_id(rs[i], is_user_test)].push_back(rs[i]);
				}

				uint rest = rs.size() - n * loop;
				for (uint j = 0; j < rest; ++j, ++i) {
					chunks[random()][get_id(rs[i], is_user_test)].push_back(rs[i]);
				}
			}
		}
				
		std::cout << "less than n:" << ct << std::endl;
		detail_info_->less_than_div_num_ = ct;

		const double matrix_size = src.userSize() * src.itemSize();
		detail_info_->sparsity_ = all_ct / matrix_size;
		for (auto const& c : chunks) {
			uint ct = 0;
			for (auto const& vec : c) ct += vec.size();
			detail_info_->chunk_sparsity_.push_back(ct / matrix_size);
		}

		return std::move(chunks);
	}


	template <class R, class F, class... ITs>
	auto run_parallel(F&& valid_func, uint n, ITs... iters) ->std::vector<R>
	{
		std::vector<R> result;

		const uint cpu_core_num = std::thread::hardware_concurrency();
		const uint div = std::ceil(static_cast<double>(n) / cpu_core_num);

		for (uint d = 0; d < div; ++d){
			std::vector<std::future<R>> task;

			for (uint i = d * cpu_core_num, size = d != div-1 ? (d + 1) * cpu_core_num : n; i < size; ++i, sig::impl::increment_iterator(iters...)){
				task.push_back(std::async(std::launch::async, valid_func, sig::impl::dereference_iterator(iters)...));
			}

			for (auto& t : task){
				result.push_back(t.get());
			}
		}

		return result;
	}
};



}
#endif