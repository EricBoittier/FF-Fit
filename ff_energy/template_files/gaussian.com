%nproc=4
%mem=5760MB
%chk={{KEY}}.chk
#P {{METHOD}}/{{BASIS}} scf(maxcycle=200) NoSymmetry

Gaussian input

0 1
{{XYZ_STRING}}


