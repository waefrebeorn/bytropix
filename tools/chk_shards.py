import struct, sys
for f in sys.argv[1:]:
    try:
        with open(f, 'rb') as fh:
            n = struct.unpack('<Q', fh.read(8))[0]
            hdr = eval(fh.read(min(n, 300)).decode('utf8', 'replace'))
        keys = [k for k in hdr.keys() if k != "__metadata__"]
        print(f, "hdrlen=", n, "ntensors=", len(keys), "first=", keys[0] if keys else None)
    except Exception as e:
        print(f, "ERROR", repr(e))
