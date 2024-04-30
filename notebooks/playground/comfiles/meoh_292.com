%nproc=4
%mem=5760MB
%chk=meoh_292.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4162 -0.0042 -0.0105
C 0.0294 0.0020 0.0049
H 1.8216 0.8846 0.0687
H -0.4066 -0.0254 -0.9938
H -0.3793 -0.8873 0.4846
H -0.4073 0.8557 0.5232

