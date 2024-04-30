%nproc=4
%mem=5760MB
%chk=meoh_735.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4322 0.0834 0.0449
C 0.0115 -0.0043 0.0042
H 1.6983 -0.3591 -0.7881
H -0.2643 -0.8510 -0.6244
H -0.4077 -0.1939 0.9923
H -0.4379 0.9165 -0.3675

