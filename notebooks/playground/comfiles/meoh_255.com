%nproc=4
%mem=5760MB
%chk=meoh_255.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4368 0.0579 -0.0704
C 0.0089 0.0033 0.0125
H 1.6727 -0.0939 0.8686
H -0.3699 -0.4385 -0.9092
H -0.2962 -0.6210 0.8523
H -0.4597 0.9800 0.1329

