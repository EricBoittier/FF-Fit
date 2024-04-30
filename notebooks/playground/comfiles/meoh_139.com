%nproc=4
%mem=5760MB
%chk=meoh_139.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4268 0.0131 0.0435
C 0.0248 -0.0028 0.0054
H 1.7202 0.6031 -0.6820
H -0.3902 -0.9719 -0.2717
H -0.4644 0.2982 0.9318
H -0.3482 0.6804 -0.7576

