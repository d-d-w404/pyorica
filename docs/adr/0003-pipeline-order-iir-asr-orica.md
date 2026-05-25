# Pipeline order: IIR → ASR → ORICA → classify → reconstruct

The pipeline runs ASR before ORICA, not after. The source repo (`receiver.py`) runs IIR → ORICA → ASR — ASR is applied to already-decomposed data as a post-processing step, with the ASR-before-ORICA code block explicitly commented out. This is a bug or incomplete refactor in the source, not an intentional design: the calibration export path in the same file (`process_calibration_data`) correctly applies IIR → ASR → ORICA.

The principled order is IIR → ASR → ORICA because ASR removes gross transient artifacts before ICA sees the data, giving ORICA cleaner input and faster convergence. ICLabel also produces more accurate labels when artifact ICs are not contaminated by residual ASR-level artifacts. This matches the standard recommended order in the EEG preprocessing literature and EEGLAB's pipeline.
