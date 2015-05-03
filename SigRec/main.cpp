#include "example/example.h"

int main()
{
	const sig::FilepathString test_folder = SIG_TO_FPSTR("C:/Users/.sigure/Documents/GitHub/Recommender/SigRec/example/test_data/");
	
	const sig::uint num_topic = 30;
	const sig::uint num_cross_validation = 5;
	const bool is_japanese_text = false;
	const bool run_pre_train = true;
	const bool make_dataset = false;

	example_ctr(test_folder, num_topic, num_cross_validation, is_japanese_text, run_pre_train, make_dataset);
	//example_mf_sgd();

	getchar();
}