%nproc=4
%mem=5760MB
%chk=meoh_107.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4469 -0.0150 0.0028
C -0.0084 0.0031 0.0057
H 1.6197 0.9438 -0.1035
H -0.3571 -1.0283 -0.0461
H -0.3725 0.5108 0.8989
H -0.2968 0.5595 -0.8861

