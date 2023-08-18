#include "colors.inc"
#include "finish.inc"

global_settings {assumed_gamma 1 max_trace_level 6}
background {color White transmit 1.0}
camera {orthographic angle 0
  right -12.24*x up 10.31*y
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
atom(<  0.06,   0.27,  -4.23>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #0
atom(< -0.23,   0.97,  -3.60>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #1
atom(<  0.69,   0.75,  -4.74>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #2
atom(< -2.78,   2.84,  -7.09>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #3
atom(< -3.62,   3.22,  -7.39>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #4
atom(< -2.54,   2.25,  -7.88>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #5
atom(<  0.75,  -2.43,  -3.62>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #6
atom(<  0.55,  -3.09,  -4.31>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #7
atom(<  0.43,  -1.59,  -3.92>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #8
atom(< -0.41,   3.21,  -5.71>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #9
atom(< -0.87,   3.13,  -4.84>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #10
atom(< -1.11,   2.94,  -6.29>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #11
atom(< -1.34,  -3.03,  -1.97>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #12
atom(< -0.64,  -3.15,  -2.66>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #13
atom(< -2.05,  -3.63,  -2.35>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #14
atom(< -4.99,  -0.10,  -3.60>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #15
atom(< -4.18,  -0.14,  -4.12>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #16
atom(< -5.70,   0.22,  -4.18>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #17
atom(<  2.83,  -0.38,  -2.91>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #18
atom(<  2.25,  -1.18,  -3.14>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #19
atom(<  2.99,  -0.09,  -3.80>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #20
atom(< -0.94,   2.83,  -3.03>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #21
atom(< -1.35,   2.48,  -2.17>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #22
atom(< -0.41,   3.57,  -2.64>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #23
atom(<  2.30,   2.09,  -7.98>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #24
atom(<  1.33,   1.90,  -8.15>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #25
atom(<  2.76,   1.72,  -8.76>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #26
atom(<  3.54,   4.02,  -6.09>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #27
atom(<  3.61,   4.79,  -6.65>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #28
atom(<  2.95,   3.45,  -6.60>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #29
atom(<  2.51,   0.25,  -5.60>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #30
atom(<  2.42,   0.98,  -6.28>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #31
atom(<  3.20,  -0.32,  -5.91>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #32
atom(<  1.37,   0.76,  -0.95>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #33
atom(<  2.13,   0.37,  -1.43>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #34
atom(<  1.51,   0.49,   0.00>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #35
atom(<  2.88,  -4.55,  -2.78>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #36
atom(<  2.02,  -4.79,  -3.22>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #37
atom(<  3.15,  -3.72,  -3.24>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #38
atom(< -2.21,   1.75,  -0.65>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #39
atom(< -1.75,   0.92,  -0.99>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #40
atom(< -3.15,   1.48,  -0.49>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #41
atom(< -4.10,  -1.92,  -1.72>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #42
atom(< -4.49,  -1.32,  -2.37>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #43
atom(< -3.17,  -1.59,  -1.67>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #44
atom(<  5.25,  -1.21,  -1.60>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #45
atom(<  4.38,  -0.97,  -1.90>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #46
atom(<  5.70,  -1.23,  -2.45>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #47
atom(< -0.82,  -1.75,  -6.39>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #48
atom(< -1.12,  -1.01,  -5.81>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #49
atom(< -1.55,  -2.39,  -6.41>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #50
atom(< -1.88,   0.72,  -8.52>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #51
atom(< -2.64,   0.14,  -8.83>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #52
atom(< -1.49,   0.29,  -7.74>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #53
atom(< -1.18,  -0.39,  -1.45>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #54
atom(< -1.23,  -1.38,  -1.48>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #55
atom(< -0.26,  -0.17,  -1.33>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #56
atom(< -2.92,   0.67,  -4.96>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #57
atom(< -3.09,   1.63,  -5.29>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #58
atom(< -2.16,   0.96,  -4.41>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #59
cylinder {<  0.06,   0.27,  -4.23>, < -0.09,   0.62,  -3.92>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.23,   0.97,  -3.60>, < -0.09,   0.62,  -3.92>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.06,   0.27,  -4.23>, <  0.37,   0.51,  -4.49>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.69,   0.75,  -4.74>, <  0.37,   0.51,  -4.49>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.78,   2.84,  -7.09>, < -3.20,   3.03,  -7.24>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -3.62,   3.22,  -7.39>, < -3.20,   3.03,  -7.24>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.78,   2.84,  -7.09>, < -2.66,   2.55,  -7.48>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.54,   2.25,  -7.88>, < -2.66,   2.55,  -7.48>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.75,  -2.43,  -3.62>, <  0.65,  -2.76,  -3.96>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.55,  -3.09,  -4.31>, <  0.65,  -2.76,  -3.96>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.75,  -2.43,  -3.62>, <  0.59,  -2.01,  -3.77>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.43,  -1.59,  -3.92>, <  0.59,  -2.01,  -3.77>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.41,   3.21,  -5.71>, < -0.64,   3.17,  -5.27>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.87,   3.13,  -4.84>, < -0.64,   3.17,  -5.27>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.41,   3.21,  -5.71>, < -0.76,   3.08,  -6.00>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.11,   2.94,  -6.29>, < -0.76,   3.08,  -6.00>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.34,  -3.03,  -1.97>, < -0.99,  -3.09,  -2.31>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.64,  -3.15,  -2.66>, < -0.99,  -3.09,  -2.31>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.34,  -3.03,  -1.97>, < -1.70,  -3.33,  -2.16>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.05,  -3.63,  -2.35>, < -1.70,  -3.33,  -2.16>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.99,  -0.10,  -3.60>, < -5.35,   0.06,  -3.89>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -5.70,   0.22,  -4.18>, < -5.35,   0.06,  -3.89>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.99,  -0.10,  -3.60>, < -4.58,  -0.12,  -3.86>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -4.18,  -0.14,  -4.12>, < -4.58,  -0.12,  -3.86>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.83,  -0.38,  -2.91>, <  2.54,  -0.78,  -3.02>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.25,  -1.18,  -3.14>, <  2.54,  -0.78,  -3.02>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.83,  -0.38,  -2.91>, <  2.91,  -0.24,  -3.35>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.99,  -0.09,  -3.80>, <  2.91,  -0.24,  -3.35>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.94,   2.83,  -3.03>, < -1.14,   2.65,  -2.60>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.35,   2.48,  -2.17>, < -1.14,   2.65,  -2.60>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.94,   2.83,  -3.03>, < -0.67,   3.20,  -2.84>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.41,   3.57,  -2.64>, < -0.67,   3.20,  -2.84>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.30,   2.09,  -7.98>, <  2.53,   1.90,  -8.37>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.76,   1.72,  -8.76>, <  2.53,   1.90,  -8.37>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.30,   2.09,  -7.98>, <  1.81,   1.99,  -8.07>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.33,   1.90,  -8.15>, <  1.81,   1.99,  -8.07>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.54,   4.02,  -6.09>, <  3.58,   4.40,  -6.37>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  3.61,   4.79,  -6.65>, <  3.58,   4.40,  -6.37>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.54,   4.02,  -6.09>, <  3.24,   3.74,  -6.34>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.95,   3.45,  -6.60>, <  3.24,   3.74,  -6.34>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.51,   0.25,  -5.60>, <  2.85,  -0.04,  -5.75>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  3.20,  -0.32,  -5.91>, <  2.85,  -0.04,  -5.75>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.51,   0.25,  -5.60>, <  2.47,   0.61,  -5.94>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.42,   0.98,  -6.28>, <  2.47,   0.61,  -5.94>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.37,   0.76,  -0.95>, <  1.44,   0.63,  -0.48>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.51,   0.49,   0.00>, <  1.44,   0.63,  -0.48>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.37,   0.76,  -0.95>, <  1.75,   0.57,  -1.19>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.13,   0.37,  -1.43>, <  1.75,   0.57,  -1.19>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.88,  -4.55,  -2.78>, <  3.01,  -4.14,  -3.01>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  3.15,  -3.72,  -3.24>, <  3.01,  -4.14,  -3.01>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.88,  -4.55,  -2.78>, <  2.45,  -4.67,  -3.00>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.02,  -4.79,  -3.22>, <  2.45,  -4.67,  -3.00>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.21,   1.75,  -0.65>, < -1.98,   1.33,  -0.82>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.75,   0.92,  -0.99>, < -1.98,   1.33,  -0.82>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.21,   1.75,  -0.65>, < -2.68,   1.61,  -0.57>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -3.15,   1.48,  -0.49>, < -2.68,   1.61,  -0.57>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.75,   0.92,  -0.99>, < -1.47,   0.26,  -1.22>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.18,  -0.39,  -1.45>, < -1.47,   0.26,  -1.22>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -4.10,  -1.92,  -1.72>, < -4.29,  -1.62,  -2.04>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -4.49,  -1.32,  -2.37>, < -4.29,  -1.62,  -2.04>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.10,  -1.92,  -1.72>, < -3.64,  -1.75,  -1.69>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -3.17,  -1.59,  -1.67>, < -3.64,  -1.75,  -1.69>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  5.25,  -1.21,  -1.60>, <  5.48,  -1.22,  -2.03>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  5.70,  -1.23,  -2.45>, <  5.48,  -1.22,  -2.03>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  5.25,  -1.21,  -1.60>, <  4.81,  -1.09,  -1.75>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  4.38,  -0.97,  -1.90>, <  4.81,  -1.09,  -1.75>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.82,  -1.75,  -6.39>, < -1.18,  -2.07,  -6.40>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.55,  -2.39,  -6.41>, < -1.18,  -2.07,  -6.40>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.82,  -1.75,  -6.39>, < -0.97,  -1.38,  -6.10>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.12,  -1.01,  -5.81>, < -0.97,  -1.38,  -6.10>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.88,   0.72,  -8.52>, < -2.26,   0.43,  -8.67>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.64,   0.14,  -8.83>, < -2.26,   0.43,  -8.67>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.88,   0.72,  -8.52>, < -1.69,   0.51,  -8.13>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.49,   0.29,  -7.74>, < -1.69,   0.51,  -8.13>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.18,  -0.39,  -1.45>, < -1.21,  -0.89,  -1.46>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.23,  -1.38,  -1.48>, < -1.21,  -0.89,  -1.46>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.18,  -0.39,  -1.45>, < -0.72,  -0.28,  -1.39>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.26,  -0.17,  -1.33>, < -0.72,  -0.28,  -1.39>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.92,   0.67,  -4.96>, < -2.54,   0.81,  -4.69>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.16,   0.96,  -4.41>, < -2.54,   0.81,  -4.69>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.92,   0.67,  -4.96>, < -3.01,   1.15,  -5.13>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -3.09,   1.63,  -5.29>, < -3.01,   1.15,  -5.13>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
// no constraints
