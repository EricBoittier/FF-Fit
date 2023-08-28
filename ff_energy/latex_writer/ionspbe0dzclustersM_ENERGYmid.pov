#include "colors.inc"
#include "finish.inc"

global_settings {assumed_gamma 1 max_trace_level 6}
background {color White transmit 1.0}
camera {orthographic angle 0
  right -9.00*x up 7.80*y
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
atom(< -1.11,   0.24,  -8.07>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #0
atom(< -1.72,  -0.56,  -8.08>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #1
atom(< -0.79,   0.26,  -7.16>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #2
atom(< -1.96,  -2.58,  -5.13>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #3
atom(< -1.54,  -1.90,  -4.60>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #4
atom(< -2.26,  -3.28,  -4.53>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #5
atom(< -1.31,   1.86,  -0.45>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #6
atom(< -1.37,   1.38,  -1.27>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #7
atom(< -0.70,   1.29,   0.00>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #8
atom(<  2.40,   0.99,  -3.17>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #9
atom(<  2.88,   1.81,  -3.47>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #10
atom(<  1.77,   0.75,  -3.91>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #11
atom(<  0.80,  -0.38,  -1.24>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #12
atom(<  1.33,  -0.01,  -1.96>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #13
atom(<  0.23,  -0.95,  -1.76>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #14
atom(<  2.60,  -2.15,  -8.23>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #15
atom(<  2.53,  -1.89,  -7.28>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #16
atom(<  2.47,  -1.24,  -8.63>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #17
atom(< -2.50,   2.25,  -6.00>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #18
atom(< -2.16,   2.69,  -6.75>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #19
atom(< -1.75,   1.73,  -5.72>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #20
atom(<  3.97,   1.23,  -6.13>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #21
atom(<  3.91,   0.51,  -5.50>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #22
atom(<  4.16,   2.06,  -5.66>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #23
atom(<  1.31,   3.45,  -5.24>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #24
atom(<  1.60,   3.28,  -6.18>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #25
atom(<  0.82,   2.59,  -5.03>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #26
atom(<  0.59,  -2.67,  -3.16>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #27
atom(<  0.98,  -2.27,  -3.96>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #28
atom(<  0.48,  -3.59,  -3.41>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #29
atom(<  2.17,  -1.84,  -5.37>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #30
atom(<  1.54,  -1.05,  -5.37>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #31
atom(<  3.03,  -1.50,  -5.05>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #32
atom(< -1.99,   0.16,  -2.68>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #33
atom(< -2.73,   0.62,  -3.07>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #34
atom(< -1.28,   0.15,  -3.35>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #35
atom(< -2.19,  -2.16,  -7.91>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #36
atom(< -2.42,  -3.09,  -8.19>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #37
atom(< -2.43,  -2.23,  -6.97>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #38
atom(<  0.12,   0.44,  -5.01>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #39
cylinder {< -1.11,   0.24,  -8.07>, < -0.95,   0.25,  -7.62>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.79,   0.26,  -7.16>, < -0.95,   0.25,  -7.62>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.11,   0.24,  -8.07>, < -1.41,  -0.16,  -8.08>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.72,  -0.56,  -8.08>, < -1.41,  -0.16,  -8.08>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.96,  -2.58,  -5.13>, < -2.11,  -2.93,  -4.83>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.26,  -3.28,  -4.53>, < -2.11,  -2.93,  -4.83>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.96,  -2.58,  -5.13>, < -1.75,  -2.24,  -4.86>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.54,  -1.90,  -4.60>, < -1.75,  -2.24,  -4.86>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.31,   1.86,  -0.45>, < -1.34,   1.62,  -0.86>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.37,   1.38,  -1.27>, < -1.34,   1.62,  -0.86>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.31,   1.86,  -0.45>, < -1.00,   1.58,  -0.22>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.70,   1.29,   0.00>, < -1.00,   1.58,  -0.22>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.40,   0.99,  -3.17>, <  2.64,   1.40,  -3.32>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.88,   1.81,  -3.47>, <  2.64,   1.40,  -3.32>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.40,   0.99,  -3.17>, <  2.08,   0.87,  -3.54>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.77,   0.75,  -3.91>, <  2.08,   0.87,  -3.54>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.77,   0.75,  -3.91>, <  0.95,   0.59,  -4.46>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.12,   0.44,  -5.01>, <  0.95,   0.59,  -4.46>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  0.80,  -0.38,  -1.24>, <  1.06,  -0.19,  -1.60>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.33,  -0.01,  -1.96>, <  1.06,  -0.19,  -1.60>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.80,  -0.38,  -1.24>, <  0.51,  -0.67,  -1.50>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.23,  -0.95,  -1.76>, <  0.51,  -0.67,  -1.50>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.60,  -2.15,  -8.23>, <  2.57,  -2.02,  -7.75>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.53,  -1.89,  -7.28>, <  2.57,  -2.02,  -7.75>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.60,  -2.15,  -8.23>, <  2.54,  -1.70,  -8.43>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.47,  -1.24,  -8.63>, <  2.54,  -1.70,  -8.43>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.50,   2.25,  -6.00>, < -2.13,   1.99,  -5.86>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.75,   1.73,  -5.72>, < -2.13,   1.99,  -5.86>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.50,   2.25,  -6.00>, < -2.33,   2.47,  -6.38>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.16,   2.69,  -6.75>, < -2.33,   2.47,  -6.38>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.97,   1.23,  -6.13>, <  4.07,   1.65,  -5.89>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  4.16,   2.06,  -5.66>, <  4.07,   1.65,  -5.89>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.97,   1.23,  -6.13>, <  3.94,   0.87,  -5.82>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  3.91,   0.51,  -5.50>, <  3.94,   0.87,  -5.82>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.31,   3.45,  -5.24>, <  1.45,   3.36,  -5.71>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.60,   3.28,  -6.18>, <  1.45,   3.36,  -5.71>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.31,   3.45,  -5.24>, <  1.06,   3.02,  -5.14>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.82,   2.59,  -5.03>, <  1.06,   3.02,  -5.14>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.59,  -2.67,  -3.16>, <  0.78,  -2.47,  -3.56>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.98,  -2.27,  -3.96>, <  0.78,  -2.47,  -3.56>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.59,  -2.67,  -3.16>, <  0.53,  -3.13,  -3.28>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.48,  -3.59,  -3.41>, <  0.53,  -3.13,  -3.28>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.17,  -1.84,  -5.37>, <  1.85,  -1.45,  -5.37>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.54,  -1.05,  -5.37>, <  1.85,  -1.45,  -5.37>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.17,  -1.84,  -5.37>, <  2.60,  -1.67,  -5.21>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  3.03,  -1.50,  -5.05>, <  2.60,  -1.67,  -5.21>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.99,   0.16,  -2.68>, < -1.64,   0.15,  -3.01>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.28,   0.15,  -3.35>, < -1.64,   0.15,  -3.01>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.99,   0.16,  -2.68>, < -2.36,   0.39,  -2.88>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.73,   0.62,  -3.07>, < -2.36,   0.39,  -2.88>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.19,  -2.16,  -7.91>, < -2.30,  -2.63,  -8.05>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.42,  -3.09,  -8.19>, < -2.30,  -2.63,  -8.05>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.19,  -2.16,  -7.91>, < -2.31,  -2.19,  -7.44>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.43,  -2.23,  -6.97>, < -2.31,  -2.19,  -7.44>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
// no constraints
