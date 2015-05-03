#pragma once

#include "../lib/sigrec.hpp"
#include "SigUtil/lib/file.hpp"
#include "SigDM/lib/ratings/sparse_boolean_matrix.hpp"

using sig::uint;
using sig::FilepathString;

#if SIG_USE_SIGNLP
#include "SigDM/lib/documents/document_loader_japanese.hpp"	// make input from Japanese documents
#include "SigDM/lib/documents/document_loader_english.hpp"	// make input from English documents

const FilepathString eng_stopword_path = L"C:/Users/.sigure/Documents/GitHub/DatasetManager/SigDM/lib/SigNLP/stopword_eng.txt";
const FilepathString TreeTagger_exe_path = SIG_TO_FPSTR("C:/Users/.sigure/Documents/TreeTagger/bin/tree-tagger.exe");
const FilepathString TreeTagger_param_path = SIG_TO_FPSTR("C:/Users/.sigure/Documents/TreeTagger/lib/english-utf8.par");

#else
#include "SigDM/lib/documents/document_loader.hpp"

#endif

/*
[ MFの使用する評価値行列データ作成 ]

・ユーザが付けた評価値データを使用
・テストデータでは、テキストの行番号がユーザIndexに対応し、各行の半角スペース区切りの値がアイテムIndexである

*/
inline auto makeBooleanRatingMatrix(
	FilepathString src_folder,
	bool make_new
)->sigdm::SparseBooleanMatrixPtr
{
	auto user_ratings = *sig::load_num2d<sig::uint>(src_folder + SIG_TO_FPSTR("user_rating.txt"), " ");
	return sigdm::SparseBooleanMatrix::makeInstance(user_ratings, true);
}



/*
[ CTRで使用する文書データ作成 ]

新規作成
・外部ファイル or プログラム内の変数から読み込む
・別途、日本語ではMeCab、英語ではTreeTaggerのインストールとパス設定が必要

過去の作成データを使用
・tokenデータ：テキスト中の各トークンに関する情報
・vocabデータ：出現単語に関する情報
*/

inline auto makeCTRData(
	bool is_japanese_text, 
	FilepathString src_folder, 
	FilepathString out_folder, 
	bool make_new
)->sigdm::DocumentSetPtr
{
	static const std::wregex url_reg(L"http(s)?://([\\w-]+\\.)+[\\w-]+(/[\\w- ./?%&=]*)?");

	const uint remove_word_frequency = 1;
	const uint remove_eng_word_length = 2;

	// 入力データ作成 
	sigdm::DocumentSetPtr inputdata;

	if (make_new) {
#if SIG_USE_SIGNLP
		if (is_japanese_text) {
			// テキストからデータセットを作成する際に使用するフィルタ設定
			sigdm::DocumentLoaderFromJapanese::FilterSetting filter(true);

			// 使用品詞の設定
			filter.addWordClass(signlp::WordClass::Noun);
			filter.addWordClass(signlp::WordClass::Adjective);
			//filter.addWordClass(signlp::WordClass::Verb);

			// 形態素解析前のフィルタ処理
			filter.setCommonPriorFilter([](sigdm::Text& str) {
				static auto& replace = sig::ZenHanReplace::get_instance();

				replace.alphabet_zen2han(str);
				replace.number_zen2han(str);
				replace.katakana_han2zen(str);
				str = std::regex_replace(str, url_reg, L"");
			});

			// 形態素解析後にフィルタ処理
			filter.setCommonPosteriorFilter([](sigdm::Text& str) {
				str = std::regex_replace(str, std::wregex(L"^\\d+$"), L"");
			});

			// 指定回数以下の出現頻度の単語は除外
			filter.setRemoveWordCount(remove_word_frequency);

			inputdata = sigdm::DocumentLoaderFromJapanese::makeInstance(src_folder, filter, out_folder);
		}
		else {
			// テキストからデータセットを作成する際に使用するフィルタ設定
			sigdm::DocumentLoaderFromEnglish::FilterSetting filter(TreeTagger_exe_path, TreeTagger_param_path, true);

			// 形態素解析前のフィルタ処理
			filter.setCommonPriorFilter([](std::wstring& str) {
				std::transform(str.begin(), str.end(), str.begin(), [](sigdm::Text::value_type e) { return isspace(e) || iscntrl(e) ? L' ' : e; });
				str = std::regex_replace(str, url_reg, L"");
				std::transform(str.begin(), str.end(), str.begin(), [](sigdm::Text::value_type e) { return tolower(e); });
			});

			const auto stopwords = *sig::load_line<FilepathString>(eng_stopword_path);

			// 形態素解析後にフィルタ処理
			filter.setCommonPosteriorFilter([&](std::wstring& str) {
				if (str.size() <= remove_eng_word_length) {
					str = sigdm::Text();
					return;
				}
				for (auto const& e : stopwords) {
					if (e == str) {
						str = sigdm::Text();
						return;
					}
				}
				str = regex_replace(str, std::wregex(L"^\\d+$"), L"");
			});

			// 指定回数以下の出現頻度の単語は除外
			filter.setRemoveWordCount(remove_word_frequency);

			inputdata = sigdm::DocumentLoaderFromEnglish::makeInstance(src_folder, filter, out_folder);
		}
#else
		assert(false);
#endif
	}
	else {
		// 過去に作成したデータセットを使用 or 自分で指定形式のデータセットを用意する場合
		inputdata = sigdm::DocumentLoader::makeInstance(out_folder, out_folder);
	}

	return inputdata;
}


inline void cleanFiles(FilepathString out_folder)
{
	auto files = sig::get_file_names(out_folder, false, SIG_TO_FPSTR(".txt"));
	
	if (files) {
		for (auto const& e : *files) {
			sig::remove_file(out_folder + e);
		}
	}
}
