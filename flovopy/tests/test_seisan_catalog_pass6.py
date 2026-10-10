"""Pass 6: enhanced event integration and waveform separation."""
import warnings
from types import SimpleNamespace

import pytest
from obspy import Stream, Trace
from obspy.core.event import Event

from flovopy.enhanced.event import EnhancedEvent, EnhancedEventMeta
from flovopy.seisanio.catalog import _enhance, to_enhanced_catalog, CatalogConversionResult
from obspy.core.event import Catalog


def test_wrap_stream_does_not_warn_and_is_not_quakeml_field():
    original = Event()
    stream = Stream([Trace()])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        enhanced = EnhancedEvent.wrap(original, meta=EnhancedEventMeta(), stream=stream)
    assert not any('attribute "stream"' in str(w.message) for w in caught)
    assert enhanced.stream is stream
    assert enhanced.resource_id == original.resource_id


def test_mvo_parser_native_conversion_is_used():
    native = EnhancedEvent.wrap(Event(), meta=EnhancedEventMeta(metrics={"aefrows": [1]}))
    parser = SimpleNamespace(to_enhancedevent=lambda: native, eventobj=Event())
    assert _enhance(parser) is native


def test_enhanced_catalog_shares_records():
    ev = EnhancedEvent.wrap(Event())
    result = CatalogConversionResult(catalog=Catalog(events=[ev]), records=[ev])
    catalog = to_enhanced_catalog(result)
    assert catalog.events[0] is catalog.records[0]
    assert catalog.events[0] is ev


def test_enhanced_catalog_requires_records():
    result = CatalogConversionResult(catalog=Catalog(events=[Event()]))
    with pytest.raises(ValueError, match="enhanced=True"):
        to_enhanced_catalog(result)
