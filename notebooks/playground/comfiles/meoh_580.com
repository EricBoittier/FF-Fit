%nproc=4
%mem=5760MB
%chk=meoh_580.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4172 0.0799 -0.0651
C 0.0271 -0.0039 0.0159
H 1.7781 -0.3108 0.7581
H -0.3143 -0.9763 0.3711
H -0.4086 0.7561 0.6645
H -0.4146 0.0941 -0.9757

