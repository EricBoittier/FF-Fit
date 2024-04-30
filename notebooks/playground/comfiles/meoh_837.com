%nproc=4
%mem=5760MB
%chk=meoh_837.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4170 0.0211 -0.0639
C 0.0350 -0.0033 0.0243
H 1.7413 0.5203 0.7147
H -0.3844 -0.0814 -0.9788
H -0.3686 -0.8557 0.5707
H -0.4387 0.9065 0.3930

