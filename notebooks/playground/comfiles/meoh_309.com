%nproc=4
%mem=5760MB
%chk=meoh_309.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4408 0.0069 0.0489
C 0.0116 -0.0018 -0.0050
H 1.5630 0.6457 -0.6843
H -0.3261 0.2750 -1.0037
H -0.3936 -0.9746 0.2738
H -0.3920 0.7513 0.6718

