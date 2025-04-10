conda activate flovopy-env
SEISAN_TOP="/data/SEISAN_DB"
DB="MVOE_"
SQLDB="${HOME}/public_html/index_mvoe.sqlite"
PATH=${PATH}:${HOME}/Developer/flovopy/flovopy/wrappers/MVOE_conversion_pipeline
python run_pipeline_a.py \
--wav_dir "${SEISAN_TOP}/WAV/${DB}" \
--aef_dir "${SEISAN_TOP}/AEF/${DB}" \
--sfile_dir "${SEISAN_TOP}/REA/${DB}" \
--archive bgs --mseed_output "${SEISAN_TOP}/miniseed/${DB}" \
--json_output "${SEISAN_TOP}/json/${DB}" \
--db ${SQLDB}

#python flovopy/wrappers/MVOE_conversion_pipeline/check_db_status2.py ${SQLDB}

# deploy
app="${HOME}/public_html/app.py"
#cp app5.py ${app}
#python ${app} # may have to kill previous app.py
#xdg-open http://localhost:8050