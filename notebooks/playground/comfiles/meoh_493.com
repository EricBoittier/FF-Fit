%nproc=4
%mem=5760MB
%chk=meoh_493.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4202 0.0117 0.0272
C 0.0204 -0.0145 0.0095
H 1.7819 0.7159 -0.5506
H -0.3847 -0.2685 0.9891
H -0.3325 0.9855 -0.2425
H -0.3915 -0.6606 -0.7657

