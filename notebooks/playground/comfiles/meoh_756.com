%nproc=4
%mem=5760MB
%chk=meoh_756.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4241 0.1083 0.0242
C 0.0153 -0.0150 -0.0063
H 1.7929 -0.6716 -0.4408
H -0.3620 -0.7597 -0.7072
H -0.3322 -0.2901 0.9895
H -0.4273 0.9668 -0.1746

