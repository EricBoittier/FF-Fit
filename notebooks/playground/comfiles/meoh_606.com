%nproc=4
%mem=5760MB
%chk=meoh_606.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4314 0.0274 -0.0719
C 0.0006 0.0085 0.0226
H 1.7277 0.3606 0.8009
H -0.2387 -1.0445 0.1707
H -0.3822 0.6307 0.8316
H -0.3762 0.3023 -0.9570

