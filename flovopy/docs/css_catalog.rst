Reading Antelope CSS3.0 catalogs
================================

Use ``flovopy.io.css`` to read fixed-width ASCII tables from a local Antelope
database volume. Supply the common basename or the path to a supported table.
The reader looks for sibling tables in the same directory. A descriptor's
external search paths are not expanded.

Read events and station picks
-----------------------------

.. code-block:: python

   from pathlib import Path
   from flovopy.io.css import read_events, read_table

   volume = Path("monthly/2009/db2009_03")
   catalog = read_events(volume)
   event = catalog[0]
   origin = event.preferred_origin()
   if origin is not None:
       print(origin.time, origin.depth)  # UTC and metres
       for arrival in origin.arrivals:
           pick = arrival.pick_id.get_referred_object()
           print(pick.waveform_id.station_code, pick.time, arrival.phase)

``CSS_Catalog`` subclasses ObsPy's Catalog. Origins with the same evid belong to
one event. The event table's prefor identifies its preferred origin by orid.
A CSS arrival becomes an ObsPy Pick. Each assoc row links a pick to an origin
and supplies the associated phase and time residual. Unassociated picks remain
in ``catalog.unassociated_picks``. Check ``catalog.issues`` for unresolved links.
Network magnitudes become ObsPy Magnitude objects. Pass a unique namespace to
``read_events(volume, namespace="avo-2009-03")`` when volumes share a basename.

Read and join individual tables
-------------------------------

.. code-block:: python

   picks = read_table(volume, "arrival")
   associations = read_table(volume.with_suffix(".assoc"))
   rso = picks.loc[picks.sta == "RSO"]
   linked = rso.merge(associations, on="arid", suffixes=("_pick", "_assoc"))
   print(linked[["time", "iphase", "phase", "orid"]])

DataFrames preserve CSS column names, epoch seconds, source units (including
origin depth in kilometres), and numeric null sentinels. Mask the appropriate
field's sentinel before calculating statistics. If a catalog is already loaded,
reuse ``catalog.tables["arrival"]`` or another loaded table.

Supported relations are event, origin, arrival, assoc, netmag, origerr,
stamag, remark, lastid, detection and detev. Missing optional tables are omitted
from ``catalog.tables``. Detector state changes are available in
``catalog.detections``; this is an empty DataFrame when detection is absent.
Use ``catalog.tables.get("detev")`` for summarized triggers. Auxiliary tables and
detector records stay in the DataFrames when exporting ObsPy events to QuakeML.
