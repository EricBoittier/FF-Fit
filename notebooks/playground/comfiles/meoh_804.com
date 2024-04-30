%nproc=4
%mem=5760MB
%chk=meoh_804.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4336 0.0893 -0.0652
C -0.0090 -0.0038 0.0225
H 1.7659 -0.4500 0.6827
H -0.2496 -0.4550 -0.9402
H -0.2913 -0.6782 0.8309
H -0.4144 0.9973 0.1687

