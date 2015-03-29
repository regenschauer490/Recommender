#include "example.h"
#include "../lib/model/ctr.h"
#include "../lib/model/lda_gibbs.h"
#include "../lib/helper/ctr_validation.hpp"

const bool ENABLE_CTR_CACHE = true;

const FilepassString rating_fname = SIG_TO_FPSTR("user");
const FilepassString theta_fname = SIG_TO_FPSTR("theta");
const FilepassString phi_fname = SIG_TO_FPSTR("phi");

void runLDA(sigtm::LDAPtr lda, FilepassString out_folder, sig::uint iteration_num)
{
	const FilepassString perp_pass = out_folder + SIG_TO_FPSTR("perplexity_ctr.txt");
	sig::clear_file(perp_pass);

	auto savePerplexity = [&](sigtm::LDA const* lda)
	{
		double perp = lda->getPerplexity();
		auto val = sig::cat(sig::split(std::to_string(perp), ","), "");
		std::cout << val << std::endl;
		sig::save_line(val, perp_pass, sig::WriteMode::append);
	};

	lda->train(iteration_num, savePerplexity);

	auto theta = lda->getTheta();
	auto phi = lda->getPhi();

	sig::save_num(theta, out_folder + theta_fname, " ");
	sig::save_num(phi, out_folder + phi_fname, " ");
}

template <class Model>
void runCTR(sigtm::CrossValidation<Model> validation, FilepassString out_folder)
{
	using namespace std;
	using sig::uint;

	out_folder = sig::modify_dirpass_tail(out_folder, true);

	//auto tmp_u = *sig::load_num2d<double>(out_folder + L"final_U.dat", " ");
	//auto tmp_v = *sig::load_num2d<double>(out_folder + L"final_V.dat", " ");
	//validation.debug_set_u(tmp_u);
	//validation.debug_set_v(tmp_v);
	
	{
		const uint N = 5;
		auto recall = validation.run(sigtm::Recall<Model>(N, sigtm::nothing));
		auto ave_pre = validation.run(sigtm::AveragePrecision<Model>(N, sigtm::nothing));
		auto cat_cov = validation.run(sigtm::CatalogueCoverage<Model>(N, sigtm::nothing));

		sig::save_num(recall, out_folder + SIG_TO_FPSTR("./recall@5.txt"), "\n");
		sig::save_num(ave_pre, out_folder + SIG_TO_FPSTR("./average_precision@5.txt"), "\n");
		sig::save_num(cat_cov, out_folder + SIG_TO_FPSTR("./catalogue_coverage@5.txt"), "\n");
	}
	{
		const uint N = 10;
		auto recall = validation.run(sigtm::Recall<Model>(N, sigtm::nothing));
		auto ave_pre = validation.run(sigtm::AveragePrecision<Model>(N, sigtm::nothing));
		auto cat_cov = validation.run(sigtm::CatalogueCoverage<Model>(N, sigtm::nothing));
		auto precision = validation.run(sigtm::Precision<Model>(N, sigtm::nothing));

		sig::save_num(precision, out_folder + SIG_TO_FPSTR("precision@10.txt"), "\n");
		sig::save_num(recall, out_folder + SIG_TO_FPSTR("./recall@10.txt"), "\n");
		sig::save_num(ave_pre, out_folder + SIG_TO_FPSTR("./average_precision@10.txt"), "\n");
		sig::save_num(cat_cov, out_folder + SIG_TO_FPSTR("./catalogue_coverage@10.txt"), "\n");
	}
	{
		const uint N = 50;
		auto recall = validation.run(sigtm::Recall<Model>(N, sigtm::nothing));
		auto ave_pre = validation.run(sigtm::AveragePrecision<Model>(N, sigtm::nothing));
		auto cat_cov = validation.run(sigtm::CatalogueCoverage<Model>(N, sigtm::nothing));

		sig::save_num(recall, out_folder + SIG_TO_FPSTR("./recall@50.txt"), "\n");
		sig::save_num(ave_pre, out_folder + SIG_TO_FPSTR("./average_precision@50.txt"), "\n");
		sig::save_num(cat_cov, out_folder + SIG_TO_FPSTR("./catalogue_coverage@50.txt"), "\n");
	}
	{
		const uint N = 100;
		auto recall = validation.run(sigtm::Recall<Model>(N, sigtm::nothing));
		auto ave_pre = validation.run(sigtm::AveragePrecision<Model>(N, sigtm::nothing));
		auto cat_cov = validation.run(sigtm::CatalogueCoverage<Model>(N, sigtm::nothing));

		sig::save_num(recall, out_folder + SIG_TO_FPSTR("./recall@100.txt"), "\n");
		sig::save_num(ave_pre, out_folder + SIG_TO_FPSTR("./average_precision@100.txt"), "\n");
		sig::save_num(cat_cov, out_folder + SIG_TO_FPSTR("./catalogue_coverage@100.txt"), "\n");
	}
	{
		const uint N = 1000;
		//auto precision = validation.run(sigtm::Precision<Model>(1000, sigtm::nothing));
		auto recall = validation.run(sigtm::Recall<Model>(N, sigtm::nothing));
		auto ave_pre = validation.run(sigtm::AveragePrecision<Model>(N, sigtm::nothing));
		auto cat_cov = validation.run(sigtm::CatalogueCoverage<Model>(N, sigtm::nothing));

		//sig::save_num(precision, L"./precision@1000.txt", "\n");
		sig::save_num(recall, out_folder + SIG_TO_FPSTR("./recall@1000.txt"), "\n");
		sig::save_num(ave_pre, out_folder + SIG_TO_FPSTR("./average_precision@1000.txt"), "\n");
		sig::save_num(cat_cov, out_folder + SIG_TO_FPSTR("./catalogue_coverage@1000.txt"), "\n");
	}
	{
		auto recall = validation.run(sigtm::Recall<Model>(sigtm::nothing, sigtm::nothing));
		auto ave_pre = validation.run(sigtm::AveragePrecision<Model>(sigtm::nothing, sigtm::nothing));
		auto cat_cov = validation.run(sigtm::CatalogueCoverage<Model>(sigtm::nothing, sigtm::nothing));

		sig::save_num(recall, out_folder + SIG_TO_FPSTR("./recall@all.txt"), "\n");
		sig::save_num(ave_pre, out_folder + SIG_TO_FPSTR("./average_precision@all.txt"), "\n");
		sig::save_num(cat_cov, out_folder + SIG_TO_FPSTR("./catalogue_coverage@all.txt"), "\n");
	}
}


void example_ctr(FilepassString src_folder, FilepassString out_folder, sig::uint topic_num, bool run_lda, bool make_new)
{
	using namespace std;

	const bool use_item_factor = true;
	const sig::uint validation_cross_num = 5;
	const sig::uint max_ctr_iteration = 50;
	const sig::uint lda_iteration_num = 300;

	src_folder = sig::modify_dirpass_tail(src_folder, true);
	out_folder = sig::modify_dirpass_tail(out_folder, true);
	if (make_new) run_lda = true;

	sigtm::DocumentLoader::PF corpus_parser = [&](sigtm::TokenList& tokens, sigtm::WordSet& words)
	{
		sigtm::DocumentLoaderSetInfo info;

		auto token_file = *sig::load_line(out_folder + sigtm::TOKEN_FILENAME);
		auto vocab_file = *sig::load_line<std::wstring>(out_folder + sigtm::VOCAB_FILENAME);

		sig::uint total_ct = 0, did = 0;
		for (auto const& line : token_file) {
			auto parsed = sig::split(line, " ");

			for (sig::uint n = 1, length = std::stoul(parsed[0]) + 1; n < length; ++n) {
				auto word_count = sig::split(parsed[n], ":");
				auto wid = std::stoul(word_count[0]);

				for (sig::uint m = 0, wct = std::stoul(word_count[1]); m < wct; ++m) {
					tokens.push_back(sigtm::Token(total_ct, did, wid));
					++total_ct;
				}
				words.emplace(wid, vocab_file[wid]);
			}
			++did;
			info.doc_names_.push_back(SIG_TO_FPSTR("doc ") + sig::to_fpstring(did));
		}

		info.doc_num_ = did;
		info.doc_type_ = sigtm::DocumentType::Defaut;
		info.is_token_sorted_ = true;
		info.working_directory_ = out_folder;

		return info;
	};

	auto docs = make_new
		? makeInputData(InputTextType::Document, src_folder, out_folder, make_new)
		: sigtm::DocumentLoader::makeInstance(corpus_parser);


	if (run_lda) {
		auto lda = sigtm::LDA_Gibbs::makeInstance(false, topic_num, docs);
		runLDA(lda, out_folder, lda_iteration_num);
	}

	auto user_ratings = *sig::load_num2d<sig::uint>(out_folder + rating_fname, " ");
	for (auto& vec : user_ratings) vec = sig::drop(1, std::move(vec));
	//auto item_ratings = *sig::load_num2d<uint>(out_folder + L"item", " ");
	//for (auto& vec : item_ratings) sig::drop(1, vec);

	auto ratings = sigtm::SparseBooleanMatrix::makeInstance(user_ratings, true);
	cout << "rating user size:" << ratings->userSize() << endl;
	cout << "rating item size:" << ratings->itemSize() << endl;

	auto hparam = sigtm::CtrHyperparameter::makeInstance(topic_num, true, ENABLE_CTR_CACHE);

	if (auto theta = sig::load_num2d<double>(out_folder + theta_fname, " ")) {
		hparam->setTheta(*theta);
	}
	if (auto beta = sig::load_num2d<double>(out_folder + phi_fname, " ")) {
		hparam->setBeta(*beta);
	}

	/*
	auto ctr = sigtm::CTR::makeInstance(TopicNum, hparam, docs, ratings);
	ctr->train(500, 10, 0);

	auto rec0 = ctr->recommend(0, true);
	std::ofstream ofs("./rec0.txt");
	for(auto const& e : rec0) ofs << e.first << " " << e.second << std::endl;

	std::vector<std::vector<double>> est(10, std::vector<double>(ratings->itemSize(), 0));
	for (uint u = 0, usize = 10; u < usize; ++u) {
		for (uint v = 0, isize = ratings->itemSize(); v < isize; ++v) {
			est[u][v] = ctr->estimate(u, v);
		}
	}

	sig::save_num(est, L"./est_ctr.txt", " ");
	*/

	sigtm::CrossValidation<sigtm::CTR> validation(validation_cross_num, use_item_factor, hparam, docs, ratings, max_ctr_iteration, 2, 0);
	runCTR<sigtm::CTR>(validation, out_folder + SIG_TO_FPSTR("validation/"));
}
