%nproc=4
%mem=5760MB
%chk=meoh_658.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4396 -0.0069 0.0056
C 0.0033 -0.0065 -0.0038
H 1.6632 0.9311 -0.1695
H -0.3977 -1.0105 -0.1430
H -0.3061 0.3507 0.9784
H -0.3904 0.6997 -0.7349

