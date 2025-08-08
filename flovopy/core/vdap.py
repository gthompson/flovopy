"""
vdap.py
--------

Module for reading legacy VDAP (Volcano Disaster Assistance Program) seismic data formats,
specifically HYPO71 earthquake catalog files and demultiplexed SUDS waveform files (DMX).

Functions:
- parse_hypo71_line: Extracts earthquake metadata from a single HYPO71 line.
- parse_hypo71_file: Reads an entire HYPO71 file into an ObsPy Catalog.
- read_DMX_file: Reads a DMX waveform file, correcting metadata and data format.
"""

import os
import numpy as np
import pandas as pd
from obspy import read, Stream, UTCDateTime
from obspy.core.event import Catalog, Event, Origin, Magnitude, Comment

def parse_hypo71_line(line):
    """
    Parses a single line of HYPO71 earthquake location output.
    Returns a dictionary of origin time, lat/lon, depth, magnitude, etc., or None if parsing fails.
    """
    try:
        year = int(line[0:2])
        month = int(line[2:4])
        day = int(line[4:6])
        hour = int(line[7:9]) if line[7:9].strip() else 0
        minute = int(line[9:11]) if line[9:11].strip() else 0
        seconds = float(line[12:17]) if line[12:17].strip() else 0

        lat_deg = int(line[17:20].strip())
        lat_min = float(line[21:26].strip())
        lat_hem = line[20].strip().upper()

        lon_deg = int(line[27:30].strip())
        lon_min = float(line[31:36].strip())
        lon_hem = line[30].strip().upper()

        depth = float(line[37:43].strip())
        magnitude = float(line[44:50].strip())
        n_ass = int(line[51:53].strip())
        time_residual = float(line[62:].strip())

        year = year + 1900 if year >= 70 else year + 2000

        add_seconds = 0
        if minute == 60:
            minute = 0
            add_seconds = 60

        origin_time = UTCDateTime(year, month, day, hour, minute, seconds) + add_seconds

        latitude = lat_deg + lat_min / 60.0
        if lat_hem == 'S':
            latitude = -latitude

        longitude = lon_deg + lon_min / 60.0
        if lon_hem == 'W':
            longitude = -longitude

        return {
            "origin_time": origin_time,
            "latitude": latitude,
            "longitude": longitude,
            "depth": depth,
            "magnitude": magnitude,
            "n_ass": n_ass,
            "time_residual": time_residual
        }
    except Exception as e:
        print(f"Failed to parse line: {line.strip()} | Error: {e}")
        return None

def parse_hypo71_file(file_path):
    """
    Parses a HYPO71 file into an ObsPy Catalog and returns unparsed lines.
    """
    catalog = Catalog()
    unparsed_lines = []
    with open(file_path, "r") as file:
        for line in file:
            event_data = parse_hypo71_line(line.strip())
            if event_data:
                event = Event()
                origin = Origin(
                    time=event_data["origin_time"],
                    latitude=event_data["latitude"],
                    longitude=event_data["longitude"],
                    depth=event_data["depth"] * 1000
                )
                magnitude = Magnitude(mag=event_data["magnitude"])

                origin.comments.append(Comment(text=f"n_ass: {event_data['n_ass']}"))
                origin.comments.append(Comment(text=f"time_residual: {event_data['time_residual']} sec"))

                event.origins.append(origin)
                event.magnitudes.append(magnitude)
                catalog.append(event)
            else:
                unparsed_lines.append(line)
    print(f"Parsed: {len(catalog)} | Unparsed: {len(unparsed_lines)}")
    return catalog, unparsed_lines

def read_DMX_file(DMXfile, fix=True, defaultnet=''):
    """
    Reads a DMX waveform file into an ObsPy Stream with metadata fixes.
    """
    print(f"Reading {DMXfile}")
    st = Stream()
    try:
        st = read(DMXfile)
        print("- read successful")
        if fix:
            for tr in st:
                if tr.stats.network == 'unk':
                    tr.stats.network = defaultnet
                tr.data = tr.data.astype(float) - 2048.0
    except Exception as e:
        print(f"- Failed to read DMX file: {e}")
    return st
