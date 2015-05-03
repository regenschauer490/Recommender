#pragma once

#include "../lib/sigrec.hpp"

void example_ctr(
	sig::FilepathString test_folder,
	sig::uint num_topic,
	sig::uint num_cross_validation,
	bool is_japanese_text,
	bool run_pre_train,
	bool make_dataset
);

void example_mf_sgd();
