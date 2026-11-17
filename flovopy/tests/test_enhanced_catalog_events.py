from obspy import UTCDateTime
from obspy.core.event import Catalog, Event, Origin

from flovopy.enhanced.catalog import EnhancedCatalog
from flovopy.enhanced.event import EnhancedEvent


def plain(seconds, event_type="earthquake"):
    event = Event(event_type=event_type)
    event.origins = [Origin(time=UTCDateTime(seconds), latitude=0, longitude=0)]
    return event


def test_one_event_list_and_plain_events_promoted_in_place():
    first, second = plain(0), plain(60, "landslide")
    catalog = EnhancedCatalog(events=[first])
    catalog.append(second)
    catalog.events.append(plain(120))           # direct list edits are seen too
    assert not hasattr(catalog, "records")
    assert [type(e) for e in catalog] == [EnhancedEvent] * 3
    assert catalog[0] is first and catalog.events[1] is second
    assert first.meta.metrics == {}
    assert len(catalog.to_dataframe()) == 3
    assert len(catalog.filter_by_event_type("landslide")) == 1
    assert len(catalog[1:]) == 2


def test_plain_obspy_catalog_events_work_in_helpers():
    source = Catalog(events=[plain(0), plain(3600)])
    catalog = EnhancedCatalog(events=source.events)
    assert catalog.events is source.events
    assert list(catalog.to_dataframe()["datetime"].notna()) == [True, True]
