%nproc=4
%mem=5760MB
%chk=meoh_989.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4195 0.0588 0.0440
C 0.0246 -0.0022 0.0220
H 1.8110 -0.0501 -0.8479
H -0.5141 0.7902 0.5417
H -0.3084 0.1149 -1.0093
H -0.3539 -0.9767 0.3305

