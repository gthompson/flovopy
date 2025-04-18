from pathlib import Path

# Define the markdown content
markdown_content = """# Extended SeisComP-like Database Schema

This document describes the schema of a custom SQLite database used to track detection, classification, amplitude-based location, and quantification of volcanic seismic events.

---

## Table: `mseed_file_status`
Tracks each cleaned MiniSEED file and its processing status.

| Column        | Type/Constraint                                  |
|---------------|--------------------------------------------------|
| `mseed_id`    | INTEGER PRIMARY KEY AUTOINCREMENT                |
| `time`        | TEXT                                             |
| `endtime`     | TEXT                                             |
| `dir`         | TEXT                                             |
| `dfile`       | TEXT UNIQUE NOT NULL                             |
| `network`     | TEXT                                             |
| `format`      | TEXT DEFAULT 'MSEED'                             |
| `detected`    | INTEGER DEFAULT 0 CHECK(detected IN (0, 1))      |
| `classified`  | INTEGER DEFAULT 0 CHECK(classified IN (0, 1))    |
| `located`     | INTEGER DEFAULT 0 CHECK(located IN (0, 1))       |
| `quantified`  | INTEGER DEFAULT 0 CHECK(quantified IN (0, 1))    |
| `comment`     | TEXT                                             |

---

## Table: `wfdisc`
Waveform file index, similar to CSS wfdisc. Unique per `dfile` and `trace_id`.

| Column        | Type/Constraint                      |
|---------------|--------------------------------------|
| `wfid`        | INTEGER PRIMARY KEY AUTOINCREMENT    |
| `trace_id`    | TEXT                                 |
| `time`        | TEXT                                 |
| `endtime`     | TEXT                                 |
| `dfile`       | TEXT                                 |
| `tracenum`    | INTEGER                              |
| `nsamp`       | INTEGER                              |
| `samprate`    | REAL                                 |
| `calib`       | REAL                                 |
| `units`       | TEXT                                 |
| `comment`     | TEXT                                 |
| UNIQUE        | (`dfile`, `trace_id`)                |

---

## Table: `network_detection`
Detection results on MiniSEED files prior to classification.

| Column             | Type/Constraint                     |
|--------------------|-------------------------------------|
| `detection_id`     | INTEGER PRIMARY KEY AUTOINCREMENT   |
| `dfile`            | TEXT                                |
| `snr`              | REAL                                |
| `minchans`         | INTEGER                             |
| `algorithm`        | TEXT                                |
| `threshon`         | REAL                                |
| `threshoff`        | REAL                                |
| `sta_seconds`      | REAL                                |
| `lta_seconds`      | REAL                                |
| `pad_seconds`      | REAL                                |
| `freq_low`         | REAL                                |
| `freq_high`        | REAL                                |
| `criterion`        | TEXT                                |
| `ontime`           | TEXT                                |
| `duration`         | TEXT                                |
| `offtime`          | TEXT                                |
| `trace_ids`        | TEXT                                |
| `detection_quality`| REAL                                |
| `comment`          | TEXT                                |

---

*This is a partial documentation excerpt. More tables (e.g., `events`, `origins`, `aef_metrics`, `trace_metrics`, `asl_grid`, etc.) exist in the full schema.*

"""

# Save the file
output_path = Path("/mnt/data/seiscomp_schema.md")
output_path.write_text(markdown_content)

# Return the file path
output_path
