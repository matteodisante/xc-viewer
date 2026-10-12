"""The trajectory pre-processing pipeline, one module per stage.

Stages (i)-(vii) of the thesis chapter "The dataset" (sec:preproc), in the order in
which they must run -- the order is argued there and is not an implementation choice::

    (i)   altitude channel      sec:altchannel   :mod:`.altchannel`
    (ii)  fix-level cleaning    sec:fixlevel     :mod:`.cleaning`
    (iii) ground-phase trimming sec:trimming     :mod:`.trimming`
    (iv)  flight-level filter   sec:flightfilter :mod:`.flightfilter`
    (v)   geographic -> ENU     sec:enu          :mod:`.enu`
    (vi)  uniform resampling    sec:uniform      :mod:`.resample`
    (vii) Savitzky-Golay        sec:savgol       :mod:`.smoothing`

Each module is a *pure transform* of a per-fix table: it takes the table a stage
receives and returns the table the next stage expects, with the per-flight record the
stage contributes to ``flights_meta``. None of them reads a file, writes a file or
chooses a data root; that is the driver script's job (``scripts/``), so a stage can be
tested against a synthetic track with known properties and nothing else.

Thresholds are never hard-coded: they live in ``configs/preprocessing.yaml`` and reach a
stage as the typed dataclasses of :mod:`xc_viewer.core.config`, whose entry
point is
``load_preproc_config``. Stage (v) is
the one stage with no thresholds at all -- it is pure coordinate geometry (thesis,
sec:enu).
"""
