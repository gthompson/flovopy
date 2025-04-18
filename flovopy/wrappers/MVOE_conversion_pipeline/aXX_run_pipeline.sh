conda activate flovopy-env
SEISAN_TOP="/data/SEISAN_DB"
DB="MVOE_"
SQLDB="${HOME}/public_html/index_mvoe4.sqlite"
PATH=${PATH}:${HOME}/Developer/flovopy/flovopy/wrappers/MVOE_conversion_pipeline
NFILES=10000000
python aXX_run_pipeline.py \
--wav_dir "${SEISAN_TOP}/WAV/${DB}" \
--aef_dir "${SEISAN_TOP}/AEF/${DB}" \
--sfile_dir "${SEISAN_TOP}/REA/${DB}" \
--archive bgs --mseed_output "${SEISAN_TOP}/miniseed/${DB}" \
--json_output "${SEISAN_TOP}/json/${DB}" \
--db ${SQLDB} \
--limit ${NFILES}
#
#python MVOE_conversion_pipeline/a06_check_index_db.py ${SQLDB}

# deploy
app="${HOME}/public_html/app.py"
#cp a07_dash_app.py ${app}
#python ${app} # may have to kill previous app.py
#xdg-open http://localhost:8050
