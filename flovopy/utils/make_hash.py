import hashlib

def make_hash(*args):
    # Convert values into one string
    base = "|".join(str(a) for a in args)
    # Use SHA256, take first 8 hex digits (short but unique enough)
    return hashlib.sha256(base.encode()).hexdigest()[:8]