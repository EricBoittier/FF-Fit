%nproc=4
%mem=5760MB
%chk=meoh_253.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4259 0.0629 -0.0656
C 0.0215 0.0013 0.0085
H 1.7328 -0.1548 0.8393
H -0.3875 -0.4559 -0.8925
H -0.3208 -0.5940 0.8550
H -0.4550 0.9759 0.1141

