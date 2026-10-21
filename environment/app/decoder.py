
import struct

def decode_redis_tape(blob: bytes):
    if not blob.startswith(b"RT"):
        raise ValueError("Invalid magic")
    num_features = struct.unpack(">H", blob[2:4])[0]
    label = blob[4]
    features = struct.unpack(">" + "f" * num_features, blob[5:])
    return label, list(features)
