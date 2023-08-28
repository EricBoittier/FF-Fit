#include "colors.inc"
#include "finish.inc"

global_settings {assumed_gamma 1 max_trace_level 6}
background {color White transmit 1.0}
camera {orthographic angle 0
  right -8.81*x up 12.00*y
  direction 50.00*z
  location <0,0,50.00> look_at <0,0,0>}


light_source {<  2.00,   3.00,  40.00> color White
  area_light <0.70, 0, 0>, <0, 0.70, 0>, 3, 3
  adaptive 1 jitter}
// no fog
#declare simple = finish {phong 0.7}
#declare pale = finish {ambient 0.5 diffuse 0.85 roughness 0.001 specular 0.200 }
#declare intermediate = finish {ambient 0.3 diffuse 0.6 specular 0.1 roughness 0.04}
#declare vmd = finish {ambient 0.0 diffuse 0.65 phong 0.1 phong_size 40.0 specular 0.5 }
#declare jmol = finish {ambient 0.2 diffuse 0.6 specular 1 roughness 0.001 metallic}
#declare ase2 = finish {ambient 0.05 brilliance 3 diffuse 0.6 metallic specular 0.7 roughness 0.04 reflection 0.15}
#declare ase3 = finish {ambient 0.15 brilliance 2 diffuse 0.6 metallic specular 1.0 roughness 0.001 reflection 0.0}
#declare glass = finish {ambient 0.05 diffuse 0.3 specular 1.0 roughness 0.001}
#declare glass2 = finish {ambient 0.01 diffuse 0.3 specular 1.0 reflection 0.25 roughness 0.001}
#declare Rcell = 0.050;
#declare Rbond = 0.100;

#macro atom(LOC, R, COL, TRANS, FIN)
  sphere{LOC, R texture{pigment{color COL transmit TRANS} finish{FIN}}}
#end
#macro constrain(LOC, R, COL, TRANS FIN)
union{torus{R, Rcell rotate 45*z texture{pigment{color COL transmit TRANS} finish{FIN}}}
     torus{R, Rcell rotate -45*z texture{pigment{color COL transmit TRANS} finish{FIN}}}
     translate LOC}
#end

// no cell vertices
atom(< -3.47,  -2.58,  -7.99>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #0
atom(< -3.03,  -3.42,  -7.90>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #1
atom(< -3.68,  -2.38,  -7.10>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #2
atom(<  0.67,   0.30,  -6.92>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #3
atom(<  0.77,   1.22,  -6.72>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #4
atom(<  0.39,   0.49,  -7.79>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #5
atom(< -0.56,  -4.91,  -6.23>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #6
atom(<  0.06,  -5.59,  -5.86>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #7
atom(< -0.63,  -5.14,  -7.19>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #8
atom(< -3.15,  -2.92,  -5.11>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #9
atom(< -2.91,  -2.98,  -4.24>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #10
atom(< -4.07,  -3.15,  -5.12>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #11
atom(< -0.87,  -1.19,  -3.42>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #12
atom(< -1.64,  -0.53,  -3.37>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #13
atom(< -1.35,  -1.92,  -3.05>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #14
atom(< -0.77,  -2.05,  -8.31>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #15
atom(< -0.56,  -1.23,  -8.66>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #16
atom(< -1.56,  -2.25,  -8.76>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #17
atom(<  1.75,  -3.94,  -9.84>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #18
atom(<  2.11,  -3.41,  -9.05>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #19
atom(<  1.57,  -4.78,  -9.29>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #20
atom(<  3.56,  -0.90,  -5.15>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #21
atom(<  3.68,  -1.45,  -4.37>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #22
atom(<  4.07,  -1.52,  -5.75>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #23
atom(<  3.90,  -3.42,  -3.55>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #24
atom(<  4.05,  -4.19,  -4.15>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #25
atom(<  2.94,  -3.45,  -3.37>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #26
atom(<  2.61,  -2.53,  -7.51>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #27
atom(<  3.34,  -2.85,  -6.85>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #28
atom(<  3.10,  -1.82,  -7.88>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #29
atom(<  2.34,   1.20,  -3.88>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #30
atom(<  3.13,   1.73,  -4.15>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #31
atom(<  2.51,   0.34,  -4.38>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #32
atom(<  1.28,  -2.96,  -3.47>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #33
atom(<  0.55,  -2.32,  -3.40>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #34
atom(<  0.91,  -3.78,  -3.11>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #35
atom(<  0.35,  -2.09,  -6.07>, 0.81, rgb <0.50, 0.20, 0.39>, 0.0, jmol) // #36
cylinder {< -3.47,  -2.58,  -7.99>, < -3.58,  -2.48,  -7.54>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -3.68,  -2.38,  -7.10>, < -3.58,  -2.48,  -7.54>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.47,  -2.58,  -7.99>, < -3.25,  -3.00,  -7.95>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -3.03,  -3.42,  -7.90>, < -3.25,  -3.00,  -7.95>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.67,   0.30,  -6.92>, <  0.53,   0.40,  -7.36>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.39,   0.49,  -7.79>, <  0.53,   0.40,  -7.36>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.67,   0.30,  -6.92>, <  0.51,  -0.90,  -6.50>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.35,  -2.09,  -6.07>, <  0.51,  -0.90,  -6.50>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {<  0.67,   0.30,  -6.92>, <  0.72,   0.76,  -6.82>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.77,   1.22,  -6.72>, <  0.72,   0.76,  -6.82>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.39,   0.49,  -7.79>, <  0.37,  -0.80,  -6.93>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.35,  -2.09,  -6.07>, <  0.37,  -0.80,  -6.93>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {< -0.56,  -4.91,  -6.23>, < -0.10,  -3.50,  -6.15>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.35,  -2.09,  -6.07>, < -0.10,  -3.50,  -6.15>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {< -0.56,  -4.91,  -6.23>, < -0.60,  -5.02,  -6.71>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.63,  -5.14,  -7.19>, < -0.60,  -5.02,  -6.71>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.56,  -4.91,  -6.23>, < -0.25,  -5.25,  -6.05>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.06,  -5.59,  -5.86>, < -0.25,  -5.25,  -6.05>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.15,  -2.92,  -5.11>, < -3.61,  -3.03,  -5.12>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -4.07,  -3.15,  -5.12>, < -3.61,  -3.03,  -5.12>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.15,  -2.92,  -5.11>, < -3.03,  -2.95,  -4.67>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.91,  -2.98,  -4.24>, < -3.03,  -2.95,  -4.67>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.87,  -1.19,  -3.42>, < -1.26,  -0.86,  -3.40>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.64,  -0.53,  -3.37>, < -1.26,  -0.86,  -3.40>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.87,  -1.19,  -3.42>, < -0.26,  -1.64,  -4.75>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.35,  -2.09,  -6.07>, < -0.26,  -1.64,  -4.75>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {< -0.87,  -1.19,  -3.42>, < -1.11,  -1.56,  -3.24>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.35,  -1.92,  -3.05>, < -1.11,  -1.56,  -3.24>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.77,  -2.05,  -8.31>, < -1.17,  -2.15,  -8.54>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.56,  -2.25,  -8.76>, < -1.17,  -2.15,  -8.54>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.77,  -2.05,  -8.31>, < -0.21,  -2.07,  -7.19>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.35,  -2.09,  -6.07>, < -0.21,  -2.07,  -7.19>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {< -0.77,  -2.05,  -8.31>, < -0.67,  -1.64,  -8.49>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.56,  -1.23,  -8.66>, < -0.67,  -1.64,  -8.49>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.56,  -1.23,  -8.66>, < -0.10,  -1.66,  -7.37>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.35,  -2.09,  -6.07>, < -0.10,  -1.66,  -7.37>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {<  1.75,  -3.94,  -9.84>, <  1.66,  -4.36,  -9.56>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.57,  -4.78,  -9.29>, <  1.66,  -4.36,  -9.56>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.75,  -3.94,  -9.84>, <  1.93,  -3.67,  -9.45>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.11,  -3.41,  -9.05>, <  1.93,  -3.67,  -9.45>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.56,  -0.90,  -5.15>, <  1.96,  -1.50,  -5.61>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.35,  -2.09,  -6.07>, <  1.96,  -1.50,  -5.61>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {<  3.56,  -0.90,  -5.15>, <  3.82,  -1.21,  -5.45>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  4.07,  -1.52,  -5.75>, <  3.82,  -1.21,  -5.45>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.56,  -0.90,  -5.15>, <  3.62,  -1.18,  -4.76>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  3.68,  -1.45,  -4.37>, <  3.62,  -1.18,  -4.76>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.90,  -3.42,  -3.55>, <  3.98,  -3.81,  -3.85>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  4.05,  -4.19,  -4.15>, <  3.98,  -3.81,  -3.85>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.90,  -3.42,  -3.55>, <  3.42,  -3.43,  -3.46>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.94,  -3.45,  -3.37>, <  3.42,  -3.43,  -3.46>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.61,  -2.53,  -7.51>, <  1.48,  -2.31,  -6.79>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.35,  -2.09,  -6.07>, <  1.48,  -2.31,  -6.79>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {<  2.61,  -2.53,  -7.51>, <  2.85,  -2.18,  -7.70>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  3.10,  -1.82,  -7.88>, <  2.85,  -2.18,  -7.70>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.61,  -2.53,  -7.51>, <  2.97,  -2.69,  -7.18>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  3.34,  -2.85,  -6.85>, <  2.97,  -2.69,  -7.18>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.34,   1.20,  -3.88>, <  2.42,   0.77,  -4.13>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.51,   0.34,  -4.38>, <  2.42,   0.77,  -4.13>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.34,   1.20,  -3.88>, <  2.73,   1.46,  -4.02>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  3.13,   1.73,  -4.15>, <  2.73,   1.46,  -4.02>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.28,  -2.96,  -3.47>, <  0.82,  -2.53,  -4.77>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.35,  -2.09,  -6.07>, <  0.82,  -2.53,  -4.77>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {<  1.28,  -2.96,  -3.47>, <  0.92,  -2.64,  -3.44>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.55,  -2.32,  -3.40>, <  0.92,  -2.64,  -3.44>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.28,  -2.96,  -3.47>, <  1.09,  -3.37,  -3.29>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.91,  -3.78,  -3.11>, <  1.09,  -3.37,  -3.29>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.55,  -2.32,  -3.40>, <  0.45,  -2.20,  -4.74>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.35,  -2.09,  -6.07>, <  0.45,  -2.20,  -4.74>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
// no constraints
