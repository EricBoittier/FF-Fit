%nproc=4
%mem=5760MB
%chk=meoh_358.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4244 0.0891 -0.0556
C 0.0410 0.0001 0.0042
H 1.6766 -0.5128 0.6756
H -0.5440 0.8107 -0.4302
H -0.3979 -0.8885 -0.4494
H -0.3739 -0.0392 1.0114

