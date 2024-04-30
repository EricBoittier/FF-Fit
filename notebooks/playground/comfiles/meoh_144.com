%nproc=4
%mem=5760MB
%chk=meoh_144.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4226 0.0241 0.0437
C 0.0154 -0.0102 0.0097
H 1.7811 0.5066 -0.7304
H -0.3742 -0.9747 -0.3161
H -0.4341 0.2775 0.9601
H -0.2784 0.7151 -0.7490

