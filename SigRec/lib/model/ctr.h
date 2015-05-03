/*
Copyright(c) 2015 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#ifndef SIGREC_CTR_HPP
#define SIGREC_CTR_HPP

#include "../sigrec.hpp"

#if SIG_USE_SIGTM

#pragma warning(disable : 4996) 
#define _SCL_SECURE_NO_WARNINGS
#define NDEBUG

#include "../sigrec.hpp"
#include "SigDM/lib/util/eigen_ublas_util.hpp"
#include "SigDM/lib/ratings/sparse_boolean_matrix.hpp"
#include "SigDM/lib/documents/document_set.hpp"
#include "SigTM/lib/model/common/lda_module.hpp"

namespace sigrec
{
using sigdm::VectorD;
using sigdm::VectorV;
using sigdm::RatingPtr;
using sigdm::SparseRatingMatrix;
using sigdm::SparseRatingMatrixPtr;
using sigdm::SparseBooleanMatrix;
using sigdm::SparseBooleanMatrixPtr; 
using sigdm::TopicId;
using sigdm::WordId;
using sigdm::TokenId;
using sigdm::TokenList;
using sigdm::TokenListPtr;
using sigdm::DocumentSetPtr;

template <class T> using MatrixUI = VectorU<VectorI<T>>;
template <class T> using MatrixIK = VectorI<VectorK<T>>;
template <class T> using MatrixKV = VectorK<VectorV<T>>;

using BlasVectorK = BlasVector<double>;
using BlasMatrixIK = BlasMatrix<double>;
using BlasMatrixUK = BlasMatrix<double>;
using BlasMatrixKK = BlasMatrix<double>;
using BlasMatrixKV = BlasMatrix<double>;
using BlasMatrixTK = BlasMatrix<double>;

struct CtrHyperparameter : boost::noncopyable
{
	VectorD<VectorK<double>>	theta_;
	VectorK<VectorV<double>>	beta_;
	uint	topic_num_;
	double	a_;				// positive update weight in U,V (the degree of effects of src ratings)
	double	b_;				// negative update weight in U,V (b < a)
	double	lambda_u_;		// penalty weight for user's feature vector
	double	lambda_v_;		// the higher lambda_v is, the more similar item's feature vector and theta is
	bool	theta_opt_;
	bool	enable_recommend_cache_;

private:
	CtrHyperparameter(uint topic_num, bool optimize_theta, bool enable_recommend_cache)
	{
		topic_num_ = topic_num;
		a_ = 1;
		b_ = 0.01;
		lambda_u_ = 0.01;
		lambda_v_ = 100;
		theta_opt_ = optimize_theta;
		enable_recommend_cache_ = enable_recommend_cache;
	}

public:
	static auto makeInstance(uint topic_num, bool optimize_theta, bool enable_recommend_cache) ->std::shared_ptr<CtrHyperparameter>{
		return std::shared_ptr<CtrHyperparameter>(new CtrHyperparameter(topic_num, optimize_theta, enable_recommend_cache));
	}

	void setTheta(std::vector<VectorK<double>> const& init){
		theta_ = init;
	}
	void setBeta(VectorK<VectorV<double>> const& init){
		beta_ = init;
	}
	void setLambdaU(double value) { lambda_u_ = value; }
	void setLambdaV(double value) { lambda_v_ = value; }
};

using CTRHyperParamPtr = std::shared_ptr<CtrHyperparameter>;


/**
\brief
	Collaborative Topic Regression model 

\details
	This model recommends each user to ranked items by topic specified collaborative filtering.
	Basic matrix factorization model for collaborative filtering learns model-parameters to estimate missing ratings from known ratings,
	however it's difficult to learn if known rating-matrix has few ratings (problem of sparseness).
	This model utilize not only ratings but also texts about items(or users), so information of texts compensate for lack of ratings.
	In detail, latent factors of parameters are affected by item(or user)'s topics extracted from texts.

	[1] Wang, C. and Blei, D.M.: Collaborative topic modeling for recommending scientific articles, Proc. ACM SIGKDD (2011)
*/
class CTR : private sigtm::impl::LDA_Module
{
public:
	using RatingValueType = int;
	using EstValueType = std::pair<Id, double>;
	using RatingPtr_ = RatingPtr<RatingValueType>;

private:
	using RatingIter_ = SparseBooleanMatrix::const_iterator;
	using RatingContainer_ = SparseBooleanMatrix::const_rating_range;
	
private:
	int const	model_id_;	// for cross validation

	CTRHyperParamPtr const	hparam_;
	DocumentSetPtr const	input_data_;
	TokenListPtr const		src_tokens_;
	TokenList const&		tokens_;
	SparseRatingMatrixPtr<RatingValueType> const ratings_;
	VectorI<std::vector<TokenId>> const			item_tokens_;	// tokens in each item(document)

	RatingContainer_ const	user_ratings_;
	RatingContainer_ const	item_ratings_;

	uint const T_;		// number of tokens
	uint const K_;		// number of topics(factor)
	uint const V_;		// number of words	
	uint const U_;		// number of users
	uint const I_;		// number of items

	BlasMatrixKV beta_;	// word distribution of topic
	BlasMatrixIK theta_;
	BlasMatrixUK user_factor_;
	BlasMatrixIK item_factor_;

	mutable Maybe<MatrixUI<Maybe<double>>>	estimate_ratings_;
	mutable Maybe<MatrixKV<double>>			term_score_;

	double likelihood_;
	double const conv_epsilon_ = 1e-4;

	// temporary
	BlasVectorK gamma_;
	BlasMatrixKV log_beta_;
	BlasMatrixKV word_ss_;
	BlasMatrixTK phi_;

private:
	void init();

	void printUFactor() const;
	void printIFactor() const;

	void saveTmp() const;
	void save() const;
	void load();

	double docInference(ItemId id, bool update_word_ss);

	void updateU();
	void updateV();
	void updateBeta();

	auto recommend_impl(Id id, bool for_user, bool ignore_train_set = true) const->std::vector<std::pair<Id, double>>;
	
private:
	CTR(CTRHyperParamPtr hparam, DocumentSetPtr docs, SparseRatingMatrixPtr<RatingValueType> ratings, int model_id)
	: model_id_(model_id), hparam_(hparam), input_data_(docs), src_tokens_(docs->getTokenList()), tokens_(*src_tokens_), ratings_(ratings), item_tokens_(docs->getDevidedDocument()),
		user_ratings_(ratings->getUsers()), item_ratings_(ratings->getItems()), 
		T_(docs->getTokenNum()), K_(hparam->topic_num_), V_(docs->getWordNum()), U_(ratings->userSize()), I_(ratings->itemSize()),
		beta_(K_, V_), theta_(I_, K_), user_factor_(U_, K_), item_factor_(I_, K_),
		estimate_ratings_(Nothing(MatrixUI<Maybe<double>>{})), term_score_(Nothing(MatrixKV<double>{})), likelihood_(-std::exp(50)),
		gamma_(K_), log_beta_(K_, V_), word_ss_(K_, V_), phi_(T_, K_)
	{
		init();
	}
	CTR(CTRHyperParamPtr hparam, DocumentSetPtr docs, SparseRatingMatrixPtr<RatingValueType> ratings)
	: CTR(hparam, docs, ratings, -1) {}
	
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
		CTRHyperParamPtr hparam,
		DocumentSetPtr docs, 
		SparseRatingMatrixPtr<RatingValueType> ratings
	) ->std::shared_ptr<CTR>
	{
		return std::shared_ptr<CTR>(new CTR(hparam, docs, ratings));
	}
	/**
	\brief
		@~japanese ファクトリ関数(Cross Validation時の並列処理用)	\n
		@~english factory function(for cross validation)	\n

	\details
		@~japanese

		@~english
	*/
	static auto makeInstance(
		CTRHyperParamPtr hparam,
		DocumentSetPtr docs, 
		SparseRatingMatrixPtr<RatingValueType> ratings,
		uint model_id
	) ->std::shared_ptr<CTR>
	{
		return std::shared_ptr<CTR>(new CTR(hparam, docs, ratings, model_id));
	}

	void train(uint max_iter, uint min_iter, Maybe<FilepathString> info_saved_dir = Nothing(), bool is_save_parameter = false);

	// return recommended item(for user) or user(for item) list (descending by estimated rating value)
	auto recommend(Id id, bool for_user, Maybe<uint> top_n, Maybe<double> threshold) const->std::vector<std::pair<Id, double>>;

	double estimate(UserId u_id, ItemId i_id) const;
	

	//ドキュメントのトピック比率
	auto getTheta() const->MatrixIK<double>{ return sigdm::impl::to_stl_matrix(theta_); }
	auto getTheta(ItemId i_id) const->VectorK<double>{ return sigdm::impl::to_stl_vector(sigdm::impl::row_(theta_, i_id)); }

	//トピックの単語比率
	auto getPhi() const->MatrixKV<double>{ return sigdm::impl::to_stl_matrix(beta_); }
	auto getPhi(TopicId k_id) const->VectorV<double>{ return sigdm::impl::to_stl_vector(sigdm::impl::row_(beta_, k_id)); }

	//トピックを強調する単語スコア
	auto getTermScore() const->MatrixKV<double>;
	auto getTermScore(TopicId t_id) const->VectorV<double>;

	// 指定トピックの上位num_get_words個の、語彙とスコアを返す
	auto getWordOfTopic(TopicId k_id, uint num_get_words, bool use_term_score = true) const->std::vector< std::tuple<std::wstring, double>>;

	uint getUserNum() const { return U_; }
	uint getItemNum() const { return I_; }
	uint getTopicNum() const { return K_; }
	uint getWordNum() const { return V_; }
	uint getUserRatingNum(uint user_id) const { return user_ratings_[user_id].size(); }
	uint getItemRatingNum(uint item_id) const { return item_ratings_[item_id].size(); }

	void debug_set_u(std::vector<std::vector<double>> const& v) {
		for (uint i = 0; i < v.size(); ++i) {
			for (uint j = 0; j < v[i].size(); ++j) user_factor_(i, j) = v[i][j];
		}
	}
	void debug_set_v(std::vector<std::vector<double>> const& v){
		for (uint i = 0; i < v.size(); ++i) {
			for (uint j = 0; j < v[i].size(); ++j)item_factor_(i, j) = v[i][j];
		}
	}
};

using CTRPtr = std::shared_ptr<CTR>;
}	// sigtm
#endif
#endif