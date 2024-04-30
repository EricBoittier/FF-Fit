%nproc=4
%mem=5760MB
%chk=meoh_526.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4285 0.0954 0.0358
C -0.0053 -0.0160 0.0031
H 1.8186 -0.4696 -0.6636
H -0.3171 -0.6759 0.8127
H -0.3407 1.0040 0.1909
H -0.3128 -0.3971 -0.9707

