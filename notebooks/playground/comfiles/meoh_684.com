%nproc=4
%mem=5760MB
%chk=meoh_684.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4312 0.0117 0.0368
C 0.0238 -0.0079 -0.0014
H 1.6834 0.6698 -0.6442
H -0.4135 -0.9557 -0.3154
H -0.4123 0.1883 0.9781
H -0.3996 0.7901 -0.6114

