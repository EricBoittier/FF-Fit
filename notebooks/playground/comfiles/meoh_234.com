%nproc=4
%mem=5760MB
%chk=meoh_234.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4392 0.1078 -0.0393
C 0.0035 -0.0112 0.0069
H 1.6802 -0.6864 0.4820
H -0.3382 -0.6246 -0.8269
H -0.3110 -0.4528 0.9525
H -0.4580 0.9715 -0.0906

