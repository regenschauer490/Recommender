/*
Copyright(c) 2014 Akihiro Nishimura

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
*/

#include "ctr.h"
#include "SigUtil/lib/tools/convergence.hpp"
#include "SigUtil/lib/functional/filter.hpp"
#include "SigUtil/lib/functional/list_deal.hpp"

#if SIG_USE_EIGEN
#include <Eigen/Dense>
#endif

#include "SigUtil/lib/tools/time_watch.hpp"

namespace sigtm
{
const double projection_z = 1.0;

static double safe_log(double x)
{
	return x > 0 ? std::log(x) : log_lower_limit;
};

template <class V>
bool is_feasible(V const& x)
{
	double val;
	double sum = 0;
	for (uint i = 0, size = x.size()-1; i < size; ++i) {
		val = x[i];
		if (val < 0 || val >1) return false;
		sum += val;
		if (sum > 1) return false;
	}
	return true;
}


// project x on to simplex (using // http://www.cs.berkeley.edu/~jduchi/projects/DuchiShSiCh08.pdf)
template <class V1, class V2>
void simplex_projection(
	V1 const& x,
	V2& x_proj,
	double z)
{
	x_proj = x;

#if SIG_USE_EIGEN
	std::sort(x_proj.data(), x_proj.data() + x_proj.size());
#else
	std::sort(x_proj.begin(), x_proj.end());
#endif

	double cumsum = -z, u;
	int j = 0;
	
	for (int i = x.size() - 1; i >= 0; --i) {
		u = x_proj[i];
		cumsum += u;
		if (u > cumsum / (j + 1)) j++;
		else break;
	}
	double theta = cumsum / j;
	for (int i = 0, size = x.size(); i < size; ++i) {
		u = x[i] - theta;
		if (u <= 0) u = 0.0;
		x_proj[i] = u;
	}
	impl::normalize_dist_v(x_proj); // fix the normaliztion issue due to numerical errors
}

template <class V1, class V2, class V3>
auto df_simplex(
	V1 const&gamma,
	V2 const& v,
	double lambda,
	V3 const& opt_x)
{
	VectorK_ g = -lambda * (opt_x - v);
	VectorK_ y = gamma;

	//sig::for_each_v([](double& v1, double v2){ v1 /= v2; }, y, opt_x);
	for(uint i = 0, size = y.size(); i < size; ++i){
		y[i] /= opt_x[i];
	}

	g += y;

#if SIG_USE_EIGEN
	g.array() *= -1;
#else
	impl::compound_assign_v([](double& v1){ v1 *= -1; }, g);
#endif

	return g;
}

template <class V1, class V2>
double f_simplex(
	V1 const& gamma,
	V2 const& v,
	double lambda,
	V2 const& opt_x)
{
	V1 y = impl::map_v([&](double x){ return safe_log(x); }, opt_x);
	V1 z = v - opt_x;

	double f = impl::inner_prod(y, gamma);	
	double val = impl::inner_prod(z, z);

	f -= 0.5 * lambda * val;

	return -f;
}

// projection gradient algorithm
template <class V1, class V2>
void optimize_simplex(
	V1 const& gamma, 
	V2 const& v, 
	double lambda,
	V2&& opt_x)
{
	size_t size = sig::min(gamma.size(), v.size());
	VectorK_ x_bar(size);
	VectorK_ opt_x_old = opt_x;

	double f_old = f_simplex(gamma, v, lambda, opt_x);

	auto g = df_simplex(gamma, v, lambda, opt_x);

	impl::normalize_dist_v(g);
	//double ab_sum = sig::sum(g);
	//if (ab_sum > 1.0) g *= (1.0 / ab_sum); // rescale the gradient

	opt_x -= g;

	simplex_projection(opt_x, x_bar, projection_z);

	x_bar -= opt_x_old;
	
	double r = 0.5 * impl::inner_prod(g, x_bar);

	const double beta = 0.5;
	double t = beta;
	for (uint iter = 0; iter < 100; ++iter) {
		opt_x = opt_x_old;
		opt_x += t * x_bar;

		double f_new = f_simplex(gamma, v, lambda, opt_x);

		if (f_new > f_old + r * t) t = t * beta;
		else break;
	}

	if (!is_feasible(opt_x))  printf("sth is wrong, not feasible. you've got to check it ...\n");
}


void CTR::init()
{
	sig::SimpleRandom<double> randf(0, 1, FixedRandom);

	if (hparam_->beta_.empty()){
		for(TopicId k = 0; k < K_; ++k){
			auto&& beta_k = impl::row_(beta_, k);
			for (ItemId v = 0; v < V_; ++v) {
				beta_k(v) = randf();
			}
			impl::normalize_dist_v(beta_k);
		}
	}
	else{
		std::cout << "beta loading" << std::endl;
		for (uint k = 0; k < K_; ++k){
			auto&& beta_k = impl::row_(beta_, k);
			for (uint v = 0; v < V_; ++v) beta_k(v) = hparam_->beta_[k][v];
			impl::normalize_dist_v(beta_k);
		}
	}

	impl::set_zero(theta_, I_, K_);

	if (hparam_->theta_opt_ && (!hparam_->theta_.empty())) {
		std::cout << "theta loading" << std::endl;
		//theta_ = sig::to_matrix_ublas(hparam_->theta_);
		for (uint i = 0; i < I_; ++i) {
			auto&& theta_i = impl::row_(theta_, i);
			for (uint k = 0; k < K_; ++k) theta_i(k) = hparam_->theta_[i][k];
		}
	}
	else {
		for (ItemId i = 0; i < I_; ++i) {
			auto&& theta_v = impl::row_(theta_, i);
			for (uint k = 0; k < K_; ++k) theta_v[k] = 0;// randf();
			//normalize_dist_v(theta_v);
		}
	}
	

	impl::set_zero(user_factor_, U_, K_);
	impl::set_zero(item_factor_, I_, K_);

	if (!hparam_->theta_opt_){
		for (ItemId i = 0; i < I_; ++i){
			auto&& if_v = impl::row_(item_factor_, i);
			for (uint k = 0; k < K_; ++k) if_v[k] = randf();
		}
	}
	else{
		item_factor_ = theta_;
	}

	//load();
}


void CTR::printUFactor() const
{
	std::cout << "user_factor" << std::endl;
	for (uint u = 0; u<U_; ++u){
		for (uint k = 0; k<K_; ++k) std::cout << user_factor_(u, k) << ", ";
		std::cout << std::endl;
	}
}
void CTR::printIFactor() const
{
	std::cout << "item_factor" << std::endl;
	for (uint i = 0; i<I_; ++i){
		for (uint k = 0; k<K_; ++k) std::cout << item_factor_(i, k) << ", ";
		std::cout << std::endl;
	}
}


const sig::FilepassString item_factor_fname = SIG_TO_FPSTR("ctr_item_factor");
const sig::FilepassString user_factor_fname = SIG_TO_FPSTR("ctr_user_factor");
const sig::FilepassString theta_fname = SIG_TO_FPSTR("ctr_theta");

template <class M>
void save_mat(sig::FilepassString pass, M const& mat) 
{
	std::ofstream ofs(pass);

	if (ofs.is_open()) {
		for (uint i = 0, size1 = impl::size_row(mat); i < size1; ++i) {
			for (uint j = 0, size2 = impl::size(impl::row_(mat, i)); j < size2; ++j) ofs << mat(i, j) << " ";
			ofs << std::endl;
		}
	}
	else std::cout << "saving file failed: " << sig::to_string(pass) << std::endl;
};

template <class M>
void load_mat(sig::FilepassString pass, M& mat)
{
	auto tmp = sig::load_num2d<double>(pass, " ");

	if (tmp) {
		for (uint i = 0, size1 = impl::size_row(mat); i < size1; ++i) {
			auto&& row = impl::row_(mat, i);
			for (uint j = 0, size2 = impl::size(row); j < size2; ++j)  row(j) = (*tmp)[i][j];
		}
		std::cout << "loading file: " << sig::to_string(pass) << std::endl;
	}
};

void CTR::save() const
{
	std::cout << "save trained parameters... ";

	auto base_pass = input_data_->getWorkingDirectory(); //+ SIG_TO_FPSTR("params/");
	auto mid = model_id_ >= 0 ? sig::to_fpstring(model_id_) : SIG_TO_FPSTR("");

	save_mat(base_pass + item_factor_fname + mid, item_factor_);
	save_mat(base_pass + user_factor_fname + mid, user_factor_);
	save_mat(base_pass + theta_fname + mid, theta_);

	std::cout << "saving file completed" << std::endl;
}

void CTR::load()
{
	//std::cout << "load prev parameters... ";

	auto base_pass = input_data_->getWorkingDirectory() + SIG_TO_FPSTR("params/");
	auto mid = model_id_ >= 0 ? sig::to_fpstring(model_id_) : SIG_TO_FPSTR("");

	load_mat(base_pass + item_factor_fname + mid, item_factor_);
	load_mat(base_pass + user_factor_fname + mid, user_factor_);
	load_mat(base_pass + theta_fname + mid, theta_);
}

double CTR::docInference(ItemId id,	bool update_word_ss)
{
	double pseudo_count = 1.0;
	double likelihood = 0;
	auto const theta_v = impl::row_(theta_, id);
	auto log_theta_v = impl::map_v([&](double x){ return safe_log(x); }, theta_v);
	
	for (auto tid : item_tokens_[id]){
		WordId w = tokens_[tid].word_id;
		auto&& phi_v = impl::row_(phi_, tid);

		for (TopicId k = 0; k < K_; ++k){
			phi_v[k] = theta_v[k] * impl::at_(beta_, k, w);
		}
		impl::normalize_dist_v(phi_v);

		for (TopicId k = 0; k < K_; ++k){
			double const& p = phi_v[k];
			if (p > 0){
				double t = log_theta_v[k];
				double lb = log_beta_(k, w);
				likelihood += p * (t + lb - std::log(p));
			}
		}
	}

	if (pseudo_count > 0) {
		likelihood += pseudo_count * impl::sum_v(log_theta_v);
	}

	// smoothing with small pseudo counts
	impl::assign_v(gamma_, pseudo_count);
	
	for (auto tid : item_tokens_[id]){
		for (TopicId k = 0; k < K_; ++k) {
			//double x = doc->m_counts[tid] * phi_(tid, k);	// doc_word_ct only
			double const& x = impl::at_(phi_, tid, k);
			gamma_[k] += x;
			
			if (update_word_ss){
				impl::at_(word_ss_, k, tokens_[tid].word_id) -= x;
			}
		}
	}

	return likelihood;
}

void CTR::updateU()
{ 
	double delta_ab = hparam_->a_ - hparam_->b_;
	MatrixKK_ XX = impl::make_zero<MatrixKK_>(K_, K_); //MatrixKK_::Zero(K_, K_);

	// calculate VCV^T in equation(8)
	for (uint i = 0; i < I_; i ++){
		if (std::begin(item_ratings_[i]) != std::end(item_ratings_[i])){
			auto const& vec_v = impl::row_(item_factor_, i);

			XX += impl::outer_prod(vec_v, vec_v);
		}
    }
	
	// negative item weight
	XX *= hparam_->b_;

#if SIG_USE_EIGEN
	XX.diagonal().array() += hparam_->lambda_u_;
#else
	sig::for_diagonal([&](double& v){ v += hparam_->lambda_u_; }, XX);
#endif

	for (uint j = 0; j < U_; ++j){
		auto const& ratings = user_ratings_[j];

		if (std::begin(ratings) != std::end(ratings)){
			MatrixKK_ A = XX;
			VectorK_ x = impl::make_zero<VectorK_>(K_);

			for (auto rating : ratings){
				auto const& vec_v = impl::row_(item_factor_, rating->item_id_);

				for (uint m = 0; m < K_; ++m) {
					for (uint n = 0; n < K_; ++n) impl::at_(A, m, n) += delta_ab * vec_v[m] * vec_v[n];
				}
				//A += delta_ab * vec_v.transpose() * vec_v;
				x += hparam_->a_ * vec_v;
			}

			auto vec_u = impl::row_(user_factor_, j);
	
			// update vector u
#if SIG_USE_EIGEN
			vec_u = A.fullPivLu().solve(x);
#else
			vec_u = *sig::matrix_vector_solve(std::move(A), std::move(x));
#endif

			// update the likelihood
			auto result = impl::inner_prod(vec_u, vec_u);
			likelihood_ += -0.5 * hparam_->lambda_u_ * result;
		}
	}
}

void CTR::updateV()
{
	double delta_ab = hparam_->a_ - hparam_->b_;
	MatrixKK_ XX = impl::make_zero<MatrixKK_>(K_, K_);
	
	for (uint j = 0; j < U_; ++j){
		if (std::begin(user_ratings_[j]) != std::end(user_ratings_[j])){
			auto const& vec_u = impl::row_(user_factor_, j);
			XX += impl::outer_prod(vec_u, vec_u);
		}
	}

#if SIG_USE_EIGEN
	XX.array() *= hparam_->b_;
#else
	impl::compound_assign_m([&](double& v) { v *= hparam_->b_; }, XX);
#endif
		
	for (uint i = 0; i < I_; ++i){
		auto&& vec_v = impl::row_(item_factor_, i);
		auto const theta_v = impl::row_(theta_, i);
		auto const& ratings = item_ratings_[i];

		if (std::begin(ratings) != std::end(ratings)){
			MatrixKK_ A = XX;
			VectorK_ x = impl::make_zero<VectorK_>(K_);

			for (auto rating : ratings){
				auto const& vec_u = impl::row_(user_factor_, rating->user_id_);

				for (uint m = 0; m < K_; ++m) {
					for (uint n = 0; n < K_; ++n) impl::at_(A, m, n) += delta_ab * vec_u[m] * vec_u[n];
				}
				//A += delta_ab * outer_prod(vec_u, vec_u);
				//A += delta_ab * vec_u.transpose() * vec_u;
				x += hparam_->a_ * vec_u;
			}

			//sig::for_each_v([&](double& x, double t){ x += hparam_->lambda_v_ * t; }, xx, theta_v);
			x += hparam_->lambda_v_ * theta_v;	// adding the topic vector
					
			MatrixKK_ B = A;		// save for computing likelihood 

			// update vector v
#if SIG_USE_EIGEN
			A.diagonal().array() += hparam_->lambda_v_;
			vec_v = A.colPivHouseholderQr().solve(x);
#else
			sig::for_diagonal([&](double& v){ v += hparam_->lambda_v_; }, A);
			vec_v = *sig::matrix_vector_solve(A, std::move(x));
#endif
			// update the likelihood for the relevant part
			likelihood_ += -0.5 * item_ratings_[i].size() * hparam_->a_;


			for (auto rating : ratings){
				auto const& vec_u = impl::row_(user_factor_, rating->user_id_);
				auto result = impl::inner_prod(vec_u, vec_u);

				likelihood_ += hparam_->a_ * result;
			}

#if SIG_USE_EIGEN
			likelihood_ += -0.5 * vec_v.dot(B * vec_v.transpose());
#else
			likelihood_ += -0.5 * impl::inner_prod(vec_v, boost::numeric::ublas::prod(B, vec_v));
#endif
			// likelihood part of theta, even when theta=0, which is a special case
			VectorK_ x2 = vec_v;
			
			//sig::for_each_v([](double& v1, double v2){ v1 -= v2; }, x2, theta_v);
			x2 -= theta_v;

			auto result = impl::inner_prod(x2, x2);
			likelihood_ += -0.5 * hparam_->lambda_v_ * result;

			if (hparam_->theta_opt_){
				likelihood_ += docInference(i, true);
				optimize_simplex(gamma_, vec_v, hparam_->lambda_v_, impl::row_(theta_, i));
			}
		}
		else{
			// m=0, this article has never been rated
			if (hparam_->theta_opt_) {
				docInference(i, false);
				impl::normalize_dist_v(gamma_);
				impl::row_(theta_, i) = gamma_;
			}
		}
	}
}

void CTR::updateBeta()
{
	beta_ = word_ss_;

	for (TopicId k = 0; k < K_; ++k){
		auto beta_v = impl::row_(beta_, k);

		impl::normalize_dist_v(beta_v);
		impl::row_(log_beta_, k) = impl::map_v([&](double x){ return safe_log(x); }, beta_v);
	}
}

auto CTR::recommend_impl(Id id, bool for_user, bool ignore_train_set) const->std::vector<EstValueType>
{
	auto get_id = [](RatingPtr_ const& rp, bool is_user) { return is_user ? rp->user_id_ : rp->item_id_; };
	auto get_estimate = [&](Id a, Id b, bool is_user) { return is_user ? estimate(a, b) : estimate(b, a); };

	std::vector<EstValueType> result;
	std::unordered_set<Id> check;

	auto&& ratings = for_user ? user_ratings_ : item_ratings_;
	const uint S = for_user ? I_ : U_;

	if (ratings.size() <= id || ratings[id].empty()) return result;

	result.reserve(S);
	if (ignore_train_set) {
		for (auto const& e : ratings[id]) check.emplace(get_id(e, !for_user));

		for (Id i = 0; i < S; ++i) {
			if(!check.count(i)) result.push_back(std::make_pair(i, get_estimate(id, i, for_user)));
		}

		/*uint i = 0;
		for (auto e : ratings[id]) {
			for (uint ed = get_id(e, !for_user); i < ed; ++i) {
				result.push_back(std::make_pair(i, get_estimates(id, i, for_user)));
			}
			i = get_id(e, !for_user) + 1;
		}
		for (; i < S; ++i) {
			result.push_back(std::make_pair(i, get_estimates(id, i, for_user)));
		}*/
	}
	else {
		for (uint i = 0; i < S; ++i) {
			result.push_back(std::make_pair(i, get_estimate(id, i, for_user)));
		}
	}
	
	sig::sort(result, [](std::pair<Id, double> const& v1, std::pair<Id, double> const& v2){ return v1.second > v2.second; });

	return result;
}

void CTR::train(uint max_iter, uint min_iter, sig::Maybe<FilepassString> info_saved_dir, bool is_save_parameter)
{
	uint iter = 0;
	double likelihood_old;
	sig::ManageConvergenceSimple conv(conv_epsilon_);

	auto base_pass = info_saved_dir ? sig::modify_dirpass_tail(*info_saved_dir, true) : input_data_->getWorkingDirectory();
	auto model_id = model_id_ >= 0 ? sig::to_fpstring(model_id_) : SIG_TO_FPSTR("");
	
	auto info_print = [&](uint iter, double likelihood, double converge){
		std::string txt = "iter=" + std::to_string(iter) + ", likelihood=" + std::to_string(likelihood) + ", converge=" + std::to_string(converge);
		std::cout << txt << std::endl;
		sig::save_line(txt, base_pass + SIG_TO_FPSTR("iteration_info.txt") + model_id, sig::WriteMode::append);
	};

	// キャッシュ用の領域確保（今後、trainの終了判定をユーザが設定できるよう変更する場合、キャッシュの再確保を行わないように変更）
	if (hparam_->enable_recommend_cache_) estimate_ratings_ = MatrixUI<Maybe<double>>(U_, std::vector<Maybe<double>>(I_, nothing));

	if (max_iter < min_iter) std::swap(max_iter, min_iter);

	if (hparam_->theta_opt_){
		gamma_ = impl::make_zero<VectorK_>(K_);
		log_beta_ = impl::map_m([&](double x){ return safe_log(x); }, beta_);
		word_ss_ = MatrixKV_(K_, V_); // SIG_INIT_MATRIX(double, K, V, 0);
		phi_ = MatrixTK_(T_, K_);  //SIG_INIT_MATRIX(double, T, K, 0);
	}
	
	// iteration until convergence
	 while (((!conv.is_convergence()) && iter < max_iter) || iter < min_iter)
	 {
		likelihood_old = likelihood_;
		likelihood_ = 0.0;

		//printUFactor();
		//printIFactor();

		//sig::TimeWatch tw;
		updateU();
		//tw.save();
		//std::cout << tw.get_total_time() << std::endl;

		//if (hparam_->lda_regression_) break; // one iteration is enough for lda-regression

		updateV();
		
		// update beta if needed
		if (hparam_->theta_opt_) updateBeta();

		//if(likelihood_ > likelihood_old) std::cout << "likelihood is decreasing!" << std::endl;
		
		++iter;
		conv.update( sig::abs_delta(likelihood_, likelihood_old) / likelihood_old);

		info_print(iter, likelihood_, conv.get_value());
	 }

	if(is_save_parameter) save();
	std::cout << "train finished" << std::endl;

	gamma_.resize(0);
	log_beta_.resize(0, 0);
	word_ss_.resize(0, 0);
	phi_.resize(0, 0);
}

auto CTR::recommend(Id id, bool for_user, sig::Maybe<uint> top_n, sig::Maybe<double> threshold) const->std::vector<EstValueType>
{
	auto result = recommend_impl(id, for_user);

	if (top_n) result = sig::take(*top_n, std::move(result));
	if (threshold) result = sig::filter([&](std::pair<Id, double> const& e){ return e.second > *threshold; }, std::move(result));

	return result;
}

inline double CTR::estimate(UserId u_id, ItemId i_id) const
{
	// todo: この判定がオーバーヘッド
	if (!estimate_ratings_) {
		auto uvec = impl::row_(user_factor_, u_id);
		auto ivec = impl::row_(item_factor_, i_id);
		return impl::inner_prod(uvec, ivec);
	}
	else if(!(*estimate_ratings_)[u_id][i_id]) {
		//return inner_prod(impl::row_(user_factor_, u_id), impl::row_(item_factor_, i_id));
		auto uvec = impl::row_(user_factor_, u_id);
		auto ivec = impl::row_(item_factor_, i_id);
		(*estimate_ratings_)[u_id][i_id] = impl::inner_prod(uvec, ivec);
	}
	return *(*estimate_ratings_)[u_id][i_id];
}

auto CTR::getTermScore() const->MatrixKV<double>
{
	using sig::operator<<=;

	if (!term_score_) {
		term_score_ <<= SIG_INIT_MATRIX(double, K, V, 0);
		calcTermScore(getPhi(), *term_score_);
	}
	return *term_score_;
}
auto CTR::getTermScore(TopicId t_id) const->VectorV<double>
{
	if (!term_score_) {
		getTermScore();
	}
	return (*term_score_)[t_id];
}

auto CTR::getWordOfTopic(TopicId k_id, uint return_word_num, bool calc_term_score) const->std::vector< std::tuple<std::wstring, double>>
{
	if (calc_term_score) return getTopWords(getTermScore(k_id), return_word_num, input_data_->words_);
	else return getTopWords(getPhi(k_id), return_word_num, input_data_->words_);
}


/*
void c_ctr::learn_map_estimate(
	const c_data* users,
	const c_data* items,
	const c_corpus* c,
	const ctr_hyperparameter* param,
	const char* directory)
{
  // init model parameters
  printf("\ninitializing the model ...\n");
  init_model(hparam_->ctr_run);

  // filename
  char name[500];

  // start time
  time_t start, current;
  time(&start);
  int elapsed = 0;

  int iter = 0;
  double likelihood = -exp(50), likelihood_old;
  double converge = 1.0;

  /// create the state log file 
  sprintf(name, "%s/state.log", directory);
  FILE* file = fopen(name, "w");
  fprintf(file, "iter time likelihood converge\n");

  int i, j, m, n, l, k;
  int* item_ids; 
  int* user_ids;

  double result;

  /// confidence parameters
  double a_minus_b = hparam_->a - hparam_->b;

  
  update();  
 
  save();

  // free memory
  gsl_matrix_free(XX);
  gsl_matrix_free(A);
  gsl_matrix_free(B);
  gsl_vector_free(x);

  if (hparam_->ctr_run && hparam_->theta_opt) {
    gsl_matrix_free(phi);
    gsl_matrix_free(log_beta);
    gsl_matrix_free(word_ss);
    gsl_vector_free(gamma);
  }
}
*/

}	// sigtm
