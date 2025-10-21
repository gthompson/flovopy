#!/bin/bash
ffmpeg -pattern_type glob  -framerate 24 -i '/home/thompsong/work/PROJECTS/SSADenver_local/AMPMAP_RESULTS/2001-??-??-????-??S.MVO*/VSAM*/map*.png'  ~/Dropbox/pdcs2001.mp4