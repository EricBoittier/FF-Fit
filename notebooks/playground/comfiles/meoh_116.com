%nproc=4
%mem=5760MB
%chk=meoh_116.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4226 -0.0045 0.0146
C 0.0233 -0.0037 0.0051
H 1.7446 0.8742 -0.2767
H -0.4077 -0.9986 -0.1074
H -0.3971 0.4417 0.9067
H -0.3390 0.5825 -0.8394

