#include "example.h"
#include "make_input.hpp"				// make inputs from dataset
#include "../lib/model/ctr.h"			// CTR model"
#include "SigTM/lib/model/lda_gibbs.h"	// LDA model for pre-training
#include "../lib/validation/ctr_validation.hpp"		// for cross validation

const bool ENABLE_CTR_CACHE = true;

void runLDA(sigtm::LDAPtr lda, FilepathString out_folder)
{
	const uint num_iteration = 500;

	const FilepathString perp_path = out_folder + SIG_TO_FPSTR("perplexity_ctr.txt");
	sig::clear_file(perp_path);

	auto savePerplexity = [&](sigtm::LDA const& lda)
	{
		double perp = lda.getPerplexity();
		auto val = sig::cat(sig::split(std::to_string(perp), ","), "");
		std::cout << "perplexity=" << val << std::endl;
		sig::save_line(val, perp_path, sig::WriteMode::append);
	};

	lda->train(num_iteration, savePerplexity);

	auto theta = lda->getTheta();
	auto phi = lda->getPhi();

	sig::save_num(theta, out_folder + SIG_TO_FPSTR("theta.dat"), " ");
	sig::save_num(phi, out_folder + SIG_TO_FPSTR("phi.dat"), " ");
}

template <class Model>
void runCV(sigrec::CrossValidation<Model> validation, FilepathString out_folder)
{
	{
		const uint N = 10;
		auto recall = validation.run(sigrec::Recall<Model>(N, sig::nothing));
		auto ave_pre = validation.run(sigrec::AveragePrecision<Model>(N, sig::nothing));
		auto cat_cov = validation.run(sigrec::CatalogueCoverage<Model>(N, sig::nothing));
		auto iud = validation.run(sigrec::InterUserDiversity<Model>(N));
		auto lpm = validation.run(sigrec::ListPersonalizationMetric<Model>(N, sig::nothing));

		sig::save_num(recall, out_folder + SIG_TO_FPSTR("./recall@10.txt"), "\n");
		sig::save_num(ave_pre, out_folder + SIG_TO_FPSTR("./average_precision@10.txt"), "\n");
		sig::save_num(cat_cov, out_folder + SIG_TO_FPSTR("./catalogue_coverage@10.txt"), "\n");
		sig::save_num(iud, out_folder + SIG_TO_FPSTR("./inter_user_diversity@10.txt"), "\n");
		sig::save_num(lpm, out_folder + SIG_TO_FPSTR("./list_personalization@10.txt"), "\n");
	}
	{
		const uint N = 50;
		auto recall = validation.run(sigrec::Recall<Model>(N, sig::nothing));
		auto ave_pre = validation.run(sigrec::AveragePrecision<Model>(N, sig::nothing));
		auto cat_cov = validation.run(sigrec::CatalogueCoverage<Model>(N, sig::nothing));
		auto iud = validation.run(sigrec::InterUserDiversity<Model>(N));
		auto lpm = validation.run(sigrec::ListPersonalizationMetric<Model>(N, sig::nothing));

		sig::save_num(recall, out_folder + SIG_TO_FPSTR("./recall@50.txt"), "\n");
		sig::save_num(ave_pre, out_folder + SIG_TO_FPSTR("./average_precision@50.txt"), "\n");
		sig::save_num(cat_cov, out_folder + SIG_TO_FPSTR("./catalogue_coverage@50.txt"), "\n");
		sig::save_num(iud, out_folder + SIG_TO_FPSTR("./inter_user_diversity@50.txt"), "\n");
		sig::save_num(lpm, out_folder + SIG_TO_FPSTR("./list_personalization@50.txt"), "\n");
	}
	{
		const uint N = 100;
		auto recall = validation.run(sigrec::Recall<Model>(N, sig::nothing));
		auto ave_pre = validation.run(sigrec::AveragePrecision<Model>(N, sig::nothing));
		auto cat_cov = validation.run(sigrec::CatalogueCoverage<Model>(N, sig::nothing));
		auto iud = validation.run(sigrec::InterUserDiversity<Model>(N));
		auto lpm = validation.run(sigrec::ListPersonalizationMetric<Model>(N, sig::nothing));

		sig::save_num(recall, out_folder + SIG_TO_FPSTR("./recall@100.txt"), "\n");
		sig::save_num(ave_pre, out_folder + SIG_TO_FPSTR("./average_precision@100.txt"), "\n");
		sig::save_num(cat_cov, out_folder + SIG_TO_FPSTR("./catalogue_coverage@100.txt"), "\n");
		sig::save_num(iud, out_folder + SIG_TO_FPSTR("./inter_user_diversity@100.txt"), "\n");
		sig::save_num(lpm, out_folder + SIG_TO_FPSTR("./list_personalization@100.txt"), "\n");
	}
	{
		auto recall = validation.run(sigrec::Recall<Model>(sig::nothing, sig::nothing));
		auto ave_pre = validation.run(sigrec::AveragePrecision<Model>(sig::nothing, sig::nothing));
		auto cat_cov = validation.run(sigrec::CatalogueCoverage<Model>(sig::nothing, sig::nothing));
		auto iud = validation.run(sigrec::InterUserDiversity<Model>(sig::nothing));
		auto lpm = validation.run(sigrec::ListPersonalizationMetric<Model>(sig::nothing, sig::nothing));

		sig::save_num(recall, out_folder + SIG_TO_FPSTR("./recall@all.txt"), "\n");
		sig::save_num(ave_pre, out_folder + SIG_TO_FPSTR("./average_precision@all.txt"), "\n");
		sig::save_num(cat_cov, out_folder + SIG_TO_FPSTR("./catalogue_coverage@all.txt"), "\n");
		sig::save_num(iud, out_folder + SIG_TO_FPSTR("./inter_user_diversity@all.txt"), "\n");
		sig::save_num(lpm, out_folder + SIG_TO_FPSTR("./list_personalization@all.txt"), "\n");
	}
}


void exp_item_factor(
	FilepathString info_folder,
	FilepathString out_folder,
	FilepathString sub_valid_folder,
	sigdm::DocumentSetPtr docs,
	sigdm::SparseBooleanMatrixPtr ratings,
	bool run_lda,
	uint num_topic,
	double lambda_u,
	double lambda_v,
	uint num_cv
){
	const bool use_item_factor = true;
	const auto out_valid_folder = sig::modify_dirpath_tail(out_folder + sub_valid_folder, true);

	const auto user_names = *sig::load_line<sigtm::Text>(info_folder + SIG_TO_FPSTR("user_list.txt"));
	const auto item_names = *sig::load_line<sigtm::Text>(info_folder + SIG_TO_FPSTR("item_list.txt"));

	if (run_lda) {
		std::cout << std::endl << "[ LDA pre-training ]" << std::endl;
		auto lda = sigtm::LDA_Gibbs::makeInstance(num_topic, docs, false);
		runLDA(lda, out_folder);
	}

	std::cout << "user size:" << ratings->userSize() << std::endl;
	std::cout << "item size:" << ratings->itemSize() << std::endl;

	auto hparam = sigrec::CtrHyperparameter::makeInstance(num_topic, true, ENABLE_CTR_CACHE);
	hparam->setLambdaU(lambda_u);
	hparam->setLambdaV(lambda_v);

	if (auto theta = sig::load_num2d<double>(out_folder + SIG_TO_FPSTR("theta.dat"), " ")) {
		hparam->setTheta(*theta);
		std::cout << "theta:" << hparam->theta_.size() << " * " << hparam->theta_[0].size() << std::endl;
	}
	if (auto beta = sig::load_num2d<double>(out_folder + SIG_TO_FPSTR("phi.dat"), " ")) {
		hparam->setBeta(*beta);
		std::cout << "phi(beta):" << hparam->beta_.size() << " * " << hparam->beta_[0].size() << std::endl;
	}

	std::cout << std::endl << "[ CTR training ]" << std::endl;
	sigrec::CrossValidation<sigrec::CTR> validation(num_cv, use_item_factor, hparam, docs, ratings, 100, 2, out_valid_folder, false);

	std::cout << std::endl << "[ Cross Validation ]" << std::endl;
	runCV<sigrec::CTR>(validation, out_valid_folder);
}

void example_ctr(
	FilepathString test_folder,
	uint num_topic,
	uint num_cross_validation,
	bool is_japanese_text,
	bool run_pre_train,
	bool make_dataset
){
	const FilepathString dataset_folder = test_folder + SIG_TO_FPSTR("dataset/item_profiles/");
	const FilepathString datainfo_folder = test_folder + SIG_TO_FPSTR("info/");
	const FilepathString out_folder = test_folder + SIG_TO_FPSTR("result/");
	const FilepathString out_cv = SIG_TO_FPSTR("validation/");

	cleanFiles(out_folder);
	cleanFiles(out_folder + out_cv);

	// 入力文書データ作成
	auto item_names = *sig::load_line<sigtm::Text>(datainfo_folder + SIG_TO_FPSTR("item_list.txt"));
	auto docs = makeCTRData(is_japanese_text, dataset_folder, out_folder, make_dataset);
	
	// 入力評価値データ作成
	auto ratings = makeBooleanRatingMatrix(datainfo_folder, make_dataset);

	exp_item_factor(
		datainfo_folder,
		out_folder,
		out_cv,
		docs,
		ratings,
		run_pre_train,
		num_topic,
		0.1,
		50,
		num_cross_validation
	);
}