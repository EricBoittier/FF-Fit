%nproc=4
%mem=5760MB
%chk=meoh_743.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4355 0.0925 0.0350
C -0.0080 -0.0061 0.0165
H 1.7388 -0.4958 -0.6878
H -0.2171 -0.7865 -0.7152
H -0.3658 -0.2605 1.0142
H -0.3875 0.9318 -0.3889

