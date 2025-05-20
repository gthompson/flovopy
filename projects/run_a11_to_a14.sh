#!/bin/bash
conda activate flovopy-env
cd ~/Developer/flovopy-test/projects
python a11_index_continous_wav_files.py > a11.log
python a12_fill_gaps_with_event_data.py > a12.log
python a13_merge_continuous.py > a13.log
python a14_preprocess_daily_sds.py > a14.log

