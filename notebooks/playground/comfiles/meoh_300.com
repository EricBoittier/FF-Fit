%nproc=4
%mem=5760MB
%chk=meoh_300.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4326 -0.0042 0.0242
C -0.0061 0.0047 -0.0089
H 1.7707 0.8482 -0.3216
H -0.3510 0.0943 -1.0390
H -0.2871 -0.9594 0.4150
H -0.3401 0.8126 0.6421

