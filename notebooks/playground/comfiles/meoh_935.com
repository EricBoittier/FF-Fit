%nproc=4
%mem=5760MB
%chk=meoh_935.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4429 0.1008 -0.0648
C -0.0073 -0.0259 0.0100
H 1.6145 -0.4271 0.7429
H -0.3930 0.9594 -0.2519
H -0.2887 -0.8415 -0.6560
H -0.2902 -0.1972 1.0486

