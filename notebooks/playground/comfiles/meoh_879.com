%nproc=4
%mem=5760MB
%chk=meoh_879.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4293 0.0162 0.0357
C 0.0224 -0.0090 0.0129
H 1.6565 0.6386 -0.6864
H -0.2991 0.3801 -0.9532
H -0.4284 -0.9971 0.1061
H -0.4240 0.6142 0.7877

