from datetime import timedelta

def floor_minute(timestamp):
    """Return the timestamp rounded down to the nearest full minute."""
    return timestamp.replace(second=0, microsecond=0)

def ceil_minute(timestamp):
    """Return the timestamp rounded up to the nearest full minute."""
    return (timestamp + timedelta(minutes=1)).replace(second=0, microsecond=0)
