%nproc=4
%mem=5760MB
%chk=meoh_670.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4250 -0.0008 0.0135
C 0.0083 0.0038 0.0166
H 1.7566 0.8217 -0.4036
H -0.2942 -0.9990 -0.2851
H -0.3845 0.2379 1.0060
H -0.3361 0.6925 -0.7549

