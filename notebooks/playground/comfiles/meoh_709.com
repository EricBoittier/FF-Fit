%nproc=4
%mem=5760MB
%chk=meoh_709.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4271 0.0418 0.0489
C 0.0285 -0.0008 0.0114
H 1.6792 0.2056 -0.8839
H -0.3051 -0.9085 -0.4915
H -0.4968 -0.0045 0.9664
H -0.4101 0.8388 -0.5279

