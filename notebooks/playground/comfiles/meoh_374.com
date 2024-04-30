%nproc=4
%mem=5760MB
%chk=meoh_374.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4231 0.0243 -0.0562
C 0.0002 -0.0029 0.0060
H 1.7914 0.4416 0.7504
H -0.3129 1.0185 -0.2105
H -0.3216 -0.7138 -0.7550
H -0.2879 -0.3122 1.0107

