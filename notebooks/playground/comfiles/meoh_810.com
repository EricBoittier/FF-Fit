%nproc=4
%mem=5760MB
%chk=meoh_810.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4298 0.0807 -0.0720
C 0.0014 -0.0097 0.0287
H 1.7476 -0.3018 0.7725
H -0.2891 -0.3587 -0.9622
H -0.3162 -0.7169 0.7949
H -0.3942 0.9960 0.1706

