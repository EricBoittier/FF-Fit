%nproc=4
%mem=5760MB
%chk=meoh_659.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4364 -0.0067 0.0067
C 0.0097 -0.0057 -0.0036
H 1.6685 0.9249 -0.1901
H -0.3995 -1.0058 -0.1467
H -0.3241 0.3400 0.9747
H -0.4003 0.7003 -0.7258

