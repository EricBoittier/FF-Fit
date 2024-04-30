%nproc=4
%mem=5760MB
%chk=meoh_701.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4291 0.0313 0.0446
C 0.0015 -0.0002 0.0107
H 1.7449 0.3645 -0.8213
H -0.2576 -0.9461 -0.4650
H -0.3785 0.0453 1.0312
H -0.3521 0.8271 -0.6047

