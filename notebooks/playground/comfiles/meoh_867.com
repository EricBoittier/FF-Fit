%nproc=4
%mem=5760MB
%chk=meoh_867.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4265 -0.0017 0.0068
C 0.0050 -0.0058 0.0101
H 1.7419 0.8882 -0.2564
H -0.3171 0.2595 -0.9969
H -0.3179 -1.0105 0.2829
H -0.3508 0.7434 0.7173

