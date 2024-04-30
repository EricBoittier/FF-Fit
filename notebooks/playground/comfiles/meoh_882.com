%nproc=4
%mem=5760MB
%chk=meoh_882.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4253 0.0213 0.0443
C 0.0389 -0.0085 0.0052
H 1.6518 0.5328 -0.7604
H -0.3919 0.4299 -0.8950
H -0.4301 -0.9894 0.0824
H -0.4565 0.5740 0.7820

