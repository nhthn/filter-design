## Nathan's Filter Design Tools

Some implementations of exotic audio filter design algorithms for NumPy (anything that is interesting or useful to me and isn't already part of NumPy/SciPy).

`polyphase_iir.py`: designs a halfband polyphase IIR filter, used for antialiasing filters in 2x upsampling or downsampling.

`hilbert_transform.py`: places poles and zeros for an analog IIR 90-degree "phase differencing network" that approximates the Hilbert transform, used in analog frequency shifter effects.

