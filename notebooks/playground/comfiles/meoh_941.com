%nproc=4
%mem=5760MB
%chk=meoh_941.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4251 0.0769 -0.0625
C -0.0065 -0.0236 0.0097
H 1.8079 -0.0842 0.8251
H -0.3094 0.9926 -0.2428
H -0.3267 -0.7305 -0.7557
H -0.2548 -0.3328 1.0250

