# wsat-decoder
Software for decoding recorded NOAA APT files.
It receives the ~15 minutes .wav file at 11025 Hz and converts it to the visible / IR image.
The demodulation process uses the Hilbert transform, basically AM demodulating the 2400 Hz carrier.
A correlation is done with the APT sync A channel (1040 Hz pulse), marking the scan lines.
Those lines are then separated and added to the figure matrix, to be plotted.
