#include "../lib/sigtm.hpp"
#include "SigUtil/lib/file.hpp"
#include "SigUtil/lib/modify/remove.hpp"
#include "SigUtil/lib/functional/list_deal.hpp"

#if SIG_USE_SIGNLP
#include "../lib/data/document_loader_japanese.hpp"
#else
#include "../lib/data/document_loader.hpp"
#endif

using sig::FilepassString;

// 入力テキストの種類 (Webページやレビュー文などの各記事はDocument, マイクロブログでの各ユーザの投稿はTweet)
enum class InputTextType { Document, Tweet };

static const std::wregex url_reg(L"http(s)?://([\\w-]+\\.)+[\\w-]+(/[\\w- ./?%&=]*)?");
static const std::wregex htag_reg(L"#(\\w)+");
static const std::wregex res_reg(L"@(\\w)+");
static const std::wregex noise_reg(L"^[ＴWＷwｗω・･、。*＊:：;；ー－…´`ﾟo｡.,_|│~~\\-\\^\"”'’＂@!！?？#⇒() () ｢」{}\\[\\]\\/ 　]+$");
static const std::wregex a_hira_kata_reg(L"^[ぁ-んァ-ン0-9０-９]$");

/*
[ 入力形式のデータ作成 ]

・新規作成(外部ファイル or プログラム内の変数)
・別途MeCabのインストールが必要

・過去の作成データを使用
・tokenデータ：テキスト中の各トークンに関する情報
・vocabデータ：出現単語に関する情報
*/

inline sigtm::DocumentSetPtr makeInputData(InputTextType tt, FilepassString src_folder, FilepassString out_folder, bool make_new)
{
	using namespace std;
	using sig::uint;

#if SIG_USE_SIGNLP
	// テキストからデータセットを作成する際に使用するフィルタ
	sigtm::DocumentLoaderFromJapanese::FilterSetting filter(true);

	// 使用品詞の設定
	filter.addWordClass(signlp::WordClass::Noun);
	filter.addWordClass(signlp::WordClass::Adjective);
	//filter.addWordClass(signlp::WordClass::Verb);

	// 形態素解析前のフィルタ処理
	filter.setCommonPriorFilter([](wstring& str) {
		//static auto& replace = sig::ZenHanReplace::get_instance();
		str = regex_replace(str, url_reg, wstring(L""));
		str = regex_replace(str, htag_reg, wstring(L""));
		str = regex_replace(str, res_reg, wstring(L""));
	});

	// 形態素解析後にフィルタ処理
	filter.setCommonPosteriorFilter([](wstring& str) {
		str = regex_replace(str, noise_reg, wstring(L""));
		str = regex_replace(str, a_hira_kata_reg, wstring(L""));
	});
#endif

	// 入力データ作成 
	sigtm::DocumentSetPtr inputdata;

	if (make_new) {
#if SIG_USE_SIGNLP
		// 新しくデータセットを作成(外部ファイルから生成)
		if (InputTextType::Tweet == tt) inputdata = sigtm::DocumentLoaderFromJapanese::makeInstanceFromTweet(src_folder, filter, out_folder);
		else inputdata = sigtm::DocumentLoaderFromJapanese::makeInstance(src_folder, filter, out_folder);
#else
		assert(false);
#endif
	}
	else {
		// 過去に作成したデータセットを使用 or 自分で指定形式のデータセットを用意する場合
		inputdata = sigtm::DocumentLoader::makeInstance(out_folder);
	}

	return inputdata;
}
