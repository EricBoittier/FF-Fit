%nproc=4
%mem=5760MB
%chk=meoh_239.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4361 0.1022 -0.0446
C -0.0123 -0.0182 0.0060
H 1.7573 -0.5723 0.5897
H -0.3321 -0.5769 -0.8736
H -0.2731 -0.4911 0.9528
H -0.3407 1.0193 -0.0575

