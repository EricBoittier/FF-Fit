%nproc=4
%mem=5760MB
%chk=meoh_384.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4327 -0.0042 -0.0304
C 0.0289 -0.0179 0.0052
H 1.5795 0.8985 0.3218
H -0.2737 1.0292 0.0026
H -0.5082 -0.4458 -0.8413
H -0.4237 -0.4164 0.9132

