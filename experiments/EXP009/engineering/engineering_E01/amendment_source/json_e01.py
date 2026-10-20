"""EXP009 E01: type-only JSON amendment. Scientific calculations are unchanged."""
import json
import numpy as np

def numpy_scalar_default(value):
    if isinstance(value,(np.integer,np.floating,np.bool_)):
        native=value.item()
        if type(native) in (int,float,bool):return native
    raise TypeError(f'Object of type {type(value).__name__} is not JSON serializable')

def dumps_e01(value):
    # Preserve V1 bytes for every already-supported payload, including identity records.
    # Unknown objects, arrays, complex values and nonfinite numbers still fail closed.
    return json.dumps(value,sort_keys=True,indent=2,allow_nan=False,default=numpy_scalar_default).encode()
