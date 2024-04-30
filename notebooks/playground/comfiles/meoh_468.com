%nproc=4
%mem=5760MB
%chk=meoh_468.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4095 0.0085 -0.0433
C 0.0374 0.0050 0.0082
H 1.8266 0.6851 0.5301
H -0.3567 0.0285 1.0242
H -0.4694 0.7862 -0.5584
H -0.3601 -0.9099 -0.4312

