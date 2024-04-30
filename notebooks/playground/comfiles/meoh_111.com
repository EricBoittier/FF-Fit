%nproc=4
%mem=5760MB
%chk=meoh_111.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4420 -0.0127 0.0074
C 0.0001 0.0008 0.0075
H 1.6339 0.9291 -0.1844
H -0.3787 -1.0177 -0.0784
H -0.3911 0.4887 0.9003
H -0.2950 0.5769 -0.8694

