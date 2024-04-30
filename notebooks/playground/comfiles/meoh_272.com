%nproc=4
%mem=5760MB
%chk=meoh_272.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4087 0.0254 -0.0451
C 0.0319 -0.0086 -0.0067
H 1.8512 0.4943 0.6930
H -0.4485 -0.2465 -0.9558
H -0.3509 -0.7290 0.7162
H -0.3340 0.9657 0.3172

