%nproc=4
%mem=5760MB
%chk=meoh_236.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4409 0.1069 -0.0422
C -0.0068 -0.0157 0.0075
H 1.6961 -0.6473 0.5292
H -0.3317 -0.6039 -0.8508
H -0.2906 -0.4683 0.9576
H -0.4046 0.9953 -0.0805

