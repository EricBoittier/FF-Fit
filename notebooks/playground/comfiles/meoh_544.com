%nproc=4
%mem=5760MB
%chk=meoh_544.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4333 0.1106 -0.0005
C -0.0169 -0.0057 0.0125
H 1.7836 -0.7912 -0.1574
H -0.1925 -0.8515 0.6773
H -0.4347 0.9431 0.3491
H -0.2472 -0.2030 -1.0345

