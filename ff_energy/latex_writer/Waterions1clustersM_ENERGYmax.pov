#include "colors.inc"
#include "finish.inc"

global_settings {assumed_gamma 1 max_trace_level 6}
background {color White transmit 1.0}
camera {orthographic angle 0
  right -7.55*x up 7.77*y
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
atom(<  1.83,  -3.20,  -6.63>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #0
atom(<  1.04,  -2.62,  -6.51>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #1
atom(<  1.79,  -3.53,  -7.52>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #2
atom(< -2.47,  -0.45,  -3.33>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #3
atom(< -2.89,  -1.02,  -3.95>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #4
atom(< -2.59,   0.38,  -3.81>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #5
atom(< -3.33,   0.41,  -0.84>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #6
atom(< -3.10,  -0.19,  -1.60>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #7
atom(< -3.09,  -0.05,  -0.02>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #8
atom(< -0.60,  -1.45,  -5.92>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #9
atom(< -0.94,  -0.77,  -6.61>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #10
atom(< -1.36,  -2.02,  -5.80>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #11
atom(<  1.79,   1.31,  -5.62>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #12
atom(<  2.49,   2.01,  -5.60>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #13
atom(<  2.29,   0.65,  -6.21>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #14
atom(<  0.36,  -1.66,  -1.52>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #15
atom(<  0.63,  -1.83,  -0.57>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #16
atom(< -0.47,  -2.12,  -1.57>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #17
atom(< -0.36,   0.65,   0.00>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #18
atom(<  0.08,   1.13,  -0.68>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #19
atom(< -0.31,  -0.25,  -0.28>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #20
atom(<  2.56,  -0.94,  -3.54>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #21
atom(<  2.52,  -1.71,  -2.87>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #22
atom(<  3.47,  -1.06,  -3.81>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #23
atom(<  1.19,   1.94,  -2.15>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #24
atom(<  1.24,   2.83,  -1.85>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #25
atom(<  1.96,   2.00,  -2.79>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #26
atom(<  0.37,   3.44,  -6.41>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #27
atom(< -0.57,   3.36,  -6.17>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #28
atom(<  0.63,   2.58,  -6.02>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #29
atom(<  2.58,  -3.18,  -2.26>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #30
atom(<  1.70,  -2.97,  -1.86>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #31
atom(<  2.32,  -3.58,  -3.12>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #32
atom(< -1.78,   2.58,  -1.86>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #33
atom(< -2.37,   1.90,  -1.53>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #34
atom(< -1.91,   2.54,  -2.84>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #35
atom(<  0.12,   0.05,  -3.79>, 0.81, rgb <0.50, 0.20, 0.39>, 0.0, jmol) // #36
atom(< -2.18,   1.89,  -5.20>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #37
cylinder {<  1.83,  -3.20,  -6.63>, <  1.44,  -2.91,  -6.57>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.04,  -2.62,  -6.51>, <  1.44,  -2.91,  -6.57>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.83,  -3.20,  -6.63>, <  1.81,  -3.37,  -7.08>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.79,  -3.53,  -7.52>, <  1.81,  -3.37,  -7.08>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.47,  -0.45,  -3.33>, < -2.68,  -0.74,  -3.64>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.89,  -1.02,  -3.95>, < -2.68,  -0.74,  -3.64>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.47,  -0.45,  -3.33>, < -1.18,  -0.20,  -3.56>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.12,   0.05,  -3.79>, < -1.18,  -0.20,  -3.56>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {< -2.47,  -0.45,  -3.33>, < -2.53,  -0.04,  -3.57>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.59,   0.38,  -3.81>, < -2.53,  -0.04,  -3.57>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.59,   0.38,  -3.81>, < -1.24,   0.21,  -3.80>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.12,   0.05,  -3.79>, < -1.24,   0.21,  -3.80>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {< -3.33,   0.41,  -0.84>, < -3.22,   0.11,  -1.22>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -3.10,  -0.19,  -1.60>, < -3.22,   0.11,  -1.22>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.33,   0.41,  -0.84>, < -3.21,   0.18,  -0.43>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -3.09,  -0.05,  -0.02>, < -3.21,   0.18,  -0.43>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.60,  -1.45,  -5.92>, < -0.98,  -1.73,  -5.86>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.36,  -2.02,  -5.80>, < -0.98,  -1.73,  -5.86>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.60,  -1.45,  -5.92>, < -0.77,  -1.11,  -6.26>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.94,  -0.77,  -6.61>, < -0.77,  -1.11,  -6.26>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.60,  -1.45,  -5.92>, < -0.24,  -0.70,  -4.85>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.12,   0.05,  -3.79>, < -0.24,  -0.70,  -4.85>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {< -0.94,  -0.77,  -6.61>, < -0.41,  -0.36,  -5.20>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.12,   0.05,  -3.79>, < -0.41,  -0.36,  -5.20>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {<  1.79,   1.31,  -5.62>, <  0.95,   0.68,  -4.71>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.12,   0.05,  -3.79>, <  0.95,   0.68,  -4.71>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {<  1.79,   1.31,  -5.62>, <  2.04,   0.98,  -5.92>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.29,   0.65,  -6.21>, <  2.04,   0.98,  -5.92>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.79,   1.31,  -5.62>, <  2.14,   1.66,  -5.61>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.49,   2.01,  -5.60>, <  2.14,   1.66,  -5.61>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.36,  -1.66,  -1.52>, <  0.24,  -0.80,  -2.65>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.12,   0.05,  -3.79>, <  0.24,  -0.80,  -2.65>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {<  0.36,  -1.66,  -1.52>, <  0.50,  -1.74,  -1.05>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.63,  -1.83,  -0.57>, <  0.50,  -1.74,  -1.05>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.36,  -1.66,  -1.52>, < -0.05,  -1.89,  -1.55>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.47,  -2.12,  -1.57>, < -0.05,  -1.89,  -1.55>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.47,  -2.12,  -1.57>, < -0.18,  -1.03,  -2.68>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.12,   0.05,  -3.79>, < -0.18,  -1.03,  -2.68>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {< -0.36,   0.65,   0.00>, < -0.34,   0.20,  -0.14>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.31,  -0.25,  -0.28>, < -0.34,   0.20,  -0.14>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.36,   0.65,   0.00>, < -0.14,   0.89,  -0.34>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.08,   1.13,  -0.68>, < -0.14,   0.89,  -0.34>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.56,  -0.94,  -3.54>, <  3.02,  -1.00,  -3.68>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  3.47,  -1.06,  -3.81>, <  3.02,  -1.00,  -3.68>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.56,  -0.94,  -3.54>, <  1.34,  -0.44,  -3.66>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.12,   0.05,  -3.79>, <  1.34,  -0.44,  -3.66>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {<  2.56,  -0.94,  -3.54>, <  2.54,  -1.32,  -3.20>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.52,  -1.71,  -2.87>, <  2.54,  -1.32,  -3.20>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.52,  -1.71,  -2.87>, <  1.32,  -0.83,  -3.33>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.12,   0.05,  -3.79>, <  1.32,  -0.83,  -3.33>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {<  2.52,  -1.71,  -2.87>, <  2.55,  -2.44,  -2.56>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.58,  -3.18,  -2.26>, <  2.55,  -2.44,  -2.56>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.19,   1.94,  -2.15>, <  0.66,   0.99,  -2.97>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.12,   0.05,  -3.79>, <  0.66,   0.99,  -2.97>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {<  1.19,   1.94,  -2.15>, <  1.58,   1.97,  -2.47>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.96,   2.00,  -2.79>, <  1.58,   1.97,  -2.47>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.19,   1.94,  -2.15>, <  1.22,   2.38,  -2.00>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.24,   2.83,  -1.85>, <  1.22,   2.38,  -2.00>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.96,   2.00,  -2.79>, <  1.04,   1.03,  -3.29>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.12,   0.05,  -3.79>, <  1.04,   1.03,  -3.29>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {<  0.37,   3.44,  -6.41>, < -0.10,   3.40,  -6.29>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.57,   3.36,  -6.17>, < -0.10,   3.40,  -6.29>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.37,   3.44,  -6.41>, <  0.50,   3.01,  -6.21>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.63,   2.58,  -6.02>, <  0.50,   3.01,  -6.21>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.58,  -3.18,  -2.26>, <  2.45,  -3.38,  -2.69>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.32,  -3.58,  -3.12>, <  2.45,  -3.38,  -2.69>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.58,  -3.18,  -2.26>, <  2.14,  -3.07,  -2.06>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.70,  -2.97,  -1.86>, <  2.14,  -3.07,  -2.06>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.78,   2.58,  -1.86>, < -1.85,   2.56,  -2.35>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.91,   2.54,  -2.84>, < -1.85,   2.56,  -2.35>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.78,   2.58,  -1.86>, < -2.07,   2.24,  -1.69>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.37,   1.90,  -1.53>, < -2.07,   2.24,  -1.69>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.12,   0.05,  -3.79>, < -1.03,   0.97,  -4.49>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {< -2.18,   1.89,  -5.20>, < -1.03,   0.97,  -4.49>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
// no constraints
