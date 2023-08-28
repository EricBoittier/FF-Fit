#include "colors.inc"
#include "finish.inc"

global_settings {assumed_gamma 1 max_trace_level 6}
background {color White transmit 1.0}
camera {orthographic angle 0
  right -10.07*x up 8.89*y
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
atom(<  0.66,  -1.96,  -5.87>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #0
atom(<  0.52,  -1.24,  -5.24>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #1
atom(<  0.95,  -1.48,  -6.71>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #2
atom(< -0.03,  -3.70,  -3.66>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #3
atom(<  0.33,  -2.87,  -4.03>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #4
atom(<  0.67,  -4.11,  -3.06>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #5
atom(<  3.29,  -0.53,  -4.09>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #6
atom(<  2.29,  -0.38,  -3.82>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #7
atom(<  3.27,  -0.66,  -5.07>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #8
atom(<  0.56,   3.35,  -2.17>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #9
atom(<  0.18,   2.72,  -2.84>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #10
atom(<  0.19,   4.11,  -2.65>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #11
atom(<  1.73,   2.27,  -5.30>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #12
atom(<  1.25,   1.78,  -4.66>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #13
atom(<  1.21,   3.04,  -5.27>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #14
atom(<  1.29,   1.19,   0.00>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #15
atom(<  1.47,   1.93,  -0.58>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #16
atom(<  0.49,   0.83,  -0.37>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #17
atom(<  2.30,  -1.50,  -1.05>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #18
atom(<  2.51,  -1.06,  -1.92>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #19
atom(<  1.34,  -1.43,  -1.09>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #20
atom(< -0.99,  -1.67,  -1.38>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #21
atom(< -0.75,  -1.14,  -2.18>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #22
atom(< -1.47,  -2.36,  -1.78>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #23
atom(< -2.20,  -1.40,  -5.08>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #24
atom(< -1.36,  -1.02,  -4.87>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #25
atom(< -2.03,  -2.36,  -5.23>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #26
atom(< -4.45,  -0.11,  -3.91>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #27
atom(< -3.58,  -0.44,  -4.14>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #28
atom(< -4.67,  -0.98,  -3.56>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #29
atom(< -1.88,   2.82,  -3.41>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #30
atom(< -1.32,   1.97,  -3.59>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #31
atom(< -2.53,   2.43,  -2.83>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #32
atom(< -1.20,   1.14,  -0.64>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #33
atom(< -2.11,   1.41,  -0.46>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #34
atom(< -1.14,   0.84,  -1.59>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #35
atom(<  0.05,   0.37,  -3.45>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #36
cylinder {<  0.66,  -1.96,  -5.87>, <  0.59,  -1.60,  -5.56>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.52,  -1.24,  -5.24>, <  0.59,  -1.60,  -5.56>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.66,  -1.96,  -5.87>, <  0.81,  -1.72,  -6.29>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.95,  -1.48,  -6.71>, <  0.81,  -1.72,  -6.29>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.03,  -3.70,  -3.66>, <  0.15,  -3.28,  -3.85>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.33,  -2.87,  -4.03>, <  0.15,  -3.28,  -3.85>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.03,  -3.70,  -3.66>, <  0.32,  -3.90,  -3.36>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.67,  -4.11,  -3.06>, <  0.32,  -3.90,  -3.36>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.29,  -0.53,  -4.09>, <  3.28,  -0.59,  -4.58>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  3.27,  -0.66,  -5.07>, <  3.28,  -0.59,  -4.58>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.29,  -0.53,  -4.09>, <  2.79,  -0.46,  -3.95>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.29,  -0.38,  -3.82>, <  2.79,  -0.46,  -3.95>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.56,   3.35,  -2.17>, <  0.37,   3.03,  -2.51>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.18,   2.72,  -2.84>, <  0.37,   3.03,  -2.51>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.56,   3.35,  -2.17>, <  0.37,   3.73,  -2.41>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.19,   4.11,  -2.65>, <  0.37,   3.73,  -2.41>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.73,   2.27,  -5.30>, <  1.49,   2.02,  -4.98>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.25,   1.78,  -4.66>, <  1.49,   2.02,  -4.98>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.73,   2.27,  -5.30>, <  1.47,   2.65,  -5.28>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.21,   3.04,  -5.27>, <  1.47,   2.65,  -5.28>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.29,   1.19,   0.00>, <  0.89,   1.01,  -0.19>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.49,   0.83,  -0.37>, <  0.89,   1.01,  -0.19>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.29,   1.19,   0.00>, <  1.38,   1.56,  -0.29>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.47,   1.93,  -0.58>, <  1.38,   1.56,  -0.29>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.30,  -1.50,  -1.05>, <  1.82,  -1.47,  -1.07>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.34,  -1.43,  -1.09>, <  1.82,  -1.47,  -1.07>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.30,  -1.50,  -1.05>, <  2.40,  -1.28,  -1.49>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.51,  -1.06,  -1.92>, <  2.40,  -1.28,  -1.49>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.99,  -1.67,  -1.38>, < -0.87,  -1.41,  -1.78>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.75,  -1.14,  -2.18>, < -0.87,  -1.41,  -1.78>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.99,  -1.67,  -1.38>, < -1.23,  -2.02,  -1.58>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.47,  -2.36,  -1.78>, < -1.23,  -2.02,  -1.58>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.20,  -1.40,  -5.08>, < -1.78,  -1.21,  -4.97>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.36,  -1.02,  -4.87>, < -1.78,  -1.21,  -4.97>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.20,  -1.40,  -5.08>, < -2.12,  -1.88,  -5.16>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.03,  -2.36,  -5.23>, < -2.12,  -1.88,  -5.16>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.45,  -0.11,  -3.91>, < -4.56,  -0.55,  -3.73>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -4.67,  -0.98,  -3.56>, < -4.56,  -0.55,  -3.73>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.45,  -0.11,  -3.91>, < -4.01,  -0.28,  -4.02>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -3.58,  -0.44,  -4.14>, < -4.01,  -0.28,  -4.02>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.88,   2.82,  -3.41>, < -2.21,   2.63,  -3.12>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.53,   2.43,  -2.83>, < -2.21,   2.63,  -3.12>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.88,   2.82,  -3.41>, < -1.60,   2.39,  -3.50>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.32,   1.97,  -3.59>, < -1.60,   2.39,  -3.50>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.20,   1.14,  -0.64>, < -1.17,   0.99,  -1.11>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.14,   0.84,  -1.59>, < -1.17,   0.99,  -1.11>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.20,   1.14,  -0.64>, < -1.66,   1.28,  -0.55>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.11,   1.41,  -0.46>, < -1.66,   1.28,  -0.55>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
// no constraints
