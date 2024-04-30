%nproc=4
%mem=5760MB
%chk=meoh_254.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4314 0.0604 -0.0682
C 0.0155 0.0024 0.0106
H 1.7013 -0.1248 0.8555
H -0.3790 -0.4471 -0.9008
H -0.3093 -0.6072 0.8538
H -0.4589 0.9772 0.1236

