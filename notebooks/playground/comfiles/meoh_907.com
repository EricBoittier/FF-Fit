%nproc=4
%mem=5760MB
%chk=meoh_907.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4217 0.1086 0.0298
C 0.0398 -0.0236 0.0157
H 1.6618 -0.5962 -0.6075
H -0.3796 0.6907 -0.6928
H -0.3434 -0.9949 -0.2970
H -0.5213 0.2427 0.9114

