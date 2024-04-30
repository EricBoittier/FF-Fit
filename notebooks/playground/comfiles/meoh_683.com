%nproc=4
%mem=5760MB
%chk=meoh_683.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4328 0.0102 0.0353
C 0.0200 -0.0067 0.0004
H 1.6805 0.6849 -0.6310
H -0.3988 -0.9622 -0.3157
H -0.4126 0.1940 0.9805
H -0.3902 0.7868 -0.6242

