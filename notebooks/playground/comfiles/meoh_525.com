%nproc=4
%mem=5760MB
%chk=meoh_525.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4301 0.0935 0.0371
C -0.0120 -0.0179 0.0037
H 1.8278 -0.4373 -0.6844
H -0.3072 -0.6664 0.8286
H -0.3185 1.0142 0.1740
H -0.3001 -0.3966 -0.9769

