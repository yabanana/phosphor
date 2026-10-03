#!/usr/bin/env python3
"""Independent controls: spatial misregistration is not temporal history."""
import json
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image
from temporal_check import measure

thresholds=json.loads((Path(__file__).parent/'testdata/temporal_thresholds.json').read_text())
with tempfile.TemporaryDirectory() as folder:
    root=Path(folder);reference=root/'reference';spatial=root/'spatial';reference.mkdir();spatial.mkdir()
    for frame in range(8):
        current=np.zeros((32,200,3),dtype=np.uint8)
        current[:,:40+frame*3]=255
        # Deliberate one-pixel spatial bias with NO access to a prior frame.
        control=np.zeros_like(current);control[:,:39+frame*3]=255
        Image.fromarray(current).save(reference/f'frame-{frame:06d}.png')
        Image.fromarray(control).save(spatial/f'frame-{frame:06d}.png')
    clean=measure(reference,spatial,thresholds,spatial_control=spatial)
    assert clean['metrics']['ghost_fraction_max']>thresholds['ghost_fraction_max']
    assert clean['metrics']['excess_ghost_fraction_max']==0
    assert clean['metrics']['support_ghost_fraction_max']==0
    assert clean['checks']['ghosting']
    # The spatial metric must also recognize a current-frame shift even
    # when its filtering differs from the independently rendered control.
    mismatched=measure(reference,spatial,thresholds,spatial_control=reference)
    assert mismatched['checks']['ghosting'],mismatched
    delayed=measure(reference,spatial,thresholds,lag=1,spatial_control=spatial)
    assert not delayed['checks']['ghosting'],delayed
    assert delayed['metrics']['excess_ghost_fraction_max']>0.25
    assert delayed['metrics']['support_ghost_fraction_max']>0.25
print('Temporal metric: no-history control distinguished from delayed-frame corruption')
