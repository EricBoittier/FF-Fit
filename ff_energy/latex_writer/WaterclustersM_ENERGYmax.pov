#include "colors.inc"
#include "finish.inc"

global_settings {assumed_gamma 1 max_trace_level 6}
background {color White transmit 1.0}
camera {orthographic angle 0
  right -11.39*x up 10.11*y
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
atom(<  0.97,   0.69,  -5.49>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #0
atom(<  0.14,   0.32,  -5.26>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #1
atom(<  1.30,   0.07,  -6.14>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #2
atom(<  1.33,  -3.60,  -8.86>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #3
atom(<  1.75,  -2.97,  -8.28>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #4
atom(<  0.39,  -3.45,  -8.62>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #5
atom(< -0.03,  -0.58,  -8.77>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #6
atom(< -0.72,  -1.03,  -8.35>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #7
atom(<  0.71,  -0.80,  -8.10>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #8
atom(< -3.02,   2.07,  -1.52>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #9
atom(< -2.61,   1.18,  -1.70>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #10
atom(< -3.41,   1.92,  -0.65>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #11
atom(< -1.67,  -0.17,  -4.09>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #12
atom(< -1.56,  -0.41,  -3.17>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #13
atom(< -1.44,  -0.97,  -4.57>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #14
atom(< -0.58,  -2.43,  -5.11>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #15
atom(< -0.75,  -3.26,  -4.57>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #16
atom(< -0.79,  -2.82,  -6.00>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #17
atom(<  2.91,  -1.01, -10.12>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #18
atom(<  3.21,  -1.73, -10.71>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #19
atom(<  2.06,  -0.75, -10.59>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #20
atom(< -4.28,   2.42,  -7.67>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #21
atom(< -4.25,   3.39,  -7.56>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #22
atom(< -3.91,   2.09,  -6.84>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #23
atom(< -4.57,  -0.72,  -2.76>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #24
atom(< -5.30,  -0.19,  -2.47>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #25
atom(< -4.09,  -0.16,  -3.36>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #26
atom(<  3.29,  -3.82,  -5.58>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #27
atom(<  2.89,  -4.69,  -5.66>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #28
atom(<  3.77,  -3.72,  -6.44>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #29
atom(< -1.25,  -3.48,  -7.68>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #30
atom(< -1.34,  -4.29,  -8.24>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #31
atom(< -2.10,  -3.01,  -7.85>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #32
atom(< -0.92,   4.11,  -5.53>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #33
atom(< -0.27,   4.69,  -5.16>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #34
atom(< -0.43,   3.62,  -6.21>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #35
atom(<  0.59,  -1.38,  -2.77>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #36
atom(<  0.27,  -1.76,  -3.62>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #37
atom(<  1.43,  -1.89,  -2.69>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #38
atom(<  4.48,   1.79,  -1.01>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #39
atom(<  4.07,   1.58,  -1.86>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #40
atom(<  3.71,   1.61,  -0.50>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #41
atom(<  0.69,   2.89,  -7.15>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #42
atom(<  0.71,   2.22,  -6.42>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #43
atom(<  1.65,   3.17,  -7.10>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #44
atom(< -2.72,  -1.16,  -7.96>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #45
atom(< -3.22,  -1.03,  -7.12>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #46
atom(< -3.31,  -0.86,  -8.65>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #47
atom(<  1.98,   2.80, -10.12>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #48
atom(<  2.86,   2.92,  -9.73>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #49
atom(<  2.08,   1.82, -10.22>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #50
atom(<  4.08,   0.63,  -5.85>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #51
atom(<  4.67,   0.25,  -6.52>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #52
atom(<  3.97,  -0.04,  -5.14>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #53
atom(<  2.02,  -0.29,  -0.35>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #54
atom(<  1.89,  -0.38,  -1.31>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #55
atom(<  1.12,  -0.21,   0.00>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #56
atom(<  4.47,  -0.81,  -3.64>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #57
atom(<  5.30,  -0.41,  -3.40>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #58
atom(<  4.58,  -1.73,  -3.45>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #59
cylinder {<  0.97,   0.69,  -5.49>, <  1.13,   0.38,  -5.81>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.30,   0.07,  -6.14>, <  1.13,   0.38,  -5.81>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.97,   0.69,  -5.49>, <  0.55,   0.51,  -5.37>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.14,   0.32,  -5.26>, <  0.55,   0.51,  -5.37>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.33,  -3.60,  -8.86>, <  1.54,  -3.29,  -8.57>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.75,  -2.97,  -8.28>, <  1.54,  -3.29,  -8.57>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.33,  -3.60,  -8.86>, <  0.86,  -3.53,  -8.74>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.39,  -3.45,  -8.62>, <  0.86,  -3.53,  -8.74>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.03,  -0.58,  -8.77>, <  0.34,  -0.69,  -8.43>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.71,  -0.80,  -8.10>, <  0.34,  -0.69,  -8.43>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.03,  -0.58,  -8.77>, < -0.37,  -0.81,  -8.56>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.72,  -1.03,  -8.35>, < -0.37,  -0.81,  -8.56>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.02,   2.07,  -1.52>, < -3.21,   1.99,  -1.09>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -3.41,   1.92,  -0.65>, < -3.21,   1.99,  -1.09>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.02,   2.07,  -1.52>, < -2.82,   1.63,  -1.61>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.61,   1.18,  -1.70>, < -2.82,   1.63,  -1.61>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.67,  -0.17,  -4.09>, < -1.56,  -0.57,  -4.33>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.44,  -0.97,  -4.57>, < -1.56,  -0.57,  -4.33>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.67,  -0.17,  -4.09>, < -1.62,  -0.29,  -3.63>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.56,  -0.41,  -3.17>, < -1.62,  -0.29,  -3.63>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.58,  -2.43,  -5.11>, < -0.68,  -2.63,  -5.56>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.79,  -2.82,  -6.00>, < -0.68,  -2.63,  -5.56>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.58,  -2.43,  -5.11>, < -0.66,  -2.85,  -4.84>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.75,  -3.26,  -4.57>, < -0.66,  -2.85,  -4.84>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.91,  -1.01, -10.12>, <  3.06,  -1.37, -10.42>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  3.21,  -1.73, -10.71>, <  3.06,  -1.37, -10.42>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.91,  -1.01, -10.12>, <  2.49,  -0.88, -10.36>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.06,  -0.75, -10.59>, <  2.49,  -0.88, -10.36>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.28,   2.42,  -7.67>, < -4.27,   2.90,  -7.61>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -4.25,   3.39,  -7.56>, < -4.27,   2.90,  -7.61>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.28,   2.42,  -7.67>, < -4.10,   2.25,  -7.25>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -3.91,   2.09,  -6.84>, < -4.10,   2.25,  -7.25>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.57,  -0.72,  -2.76>, < -4.94,  -0.46,  -2.61>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -5.30,  -0.19,  -2.47>, < -4.94,  -0.46,  -2.61>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.57,  -0.72,  -2.76>, < -4.33,  -0.44,  -3.06>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -4.09,  -0.16,  -3.36>, < -4.33,  -0.44,  -3.06>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.29,  -3.82,  -5.58>, <  3.53,  -3.77,  -6.01>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  3.77,  -3.72,  -6.44>, <  3.53,  -3.77,  -6.01>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.29,  -3.82,  -5.58>, <  3.09,  -4.25,  -5.62>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.89,  -4.69,  -5.66>, <  3.09,  -4.25,  -5.62>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.25,  -3.48,  -7.68>, < -1.29,  -3.89,  -7.96>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.34,  -4.29,  -8.24>, < -1.29,  -3.89,  -7.96>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.25,  -3.48,  -7.68>, < -1.67,  -3.25,  -7.76>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.10,  -3.01,  -7.85>, < -1.67,  -3.25,  -7.76>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.92,   4.11,  -5.53>, < -0.68,   3.86,  -5.87>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.43,   3.62,  -6.21>, < -0.68,   3.86,  -5.87>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.92,   4.11,  -5.53>, < -0.60,   4.40,  -5.34>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.27,   4.69,  -5.16>, < -0.60,   4.40,  -5.34>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.43,   3.62,  -6.21>, <  0.13,   3.25,  -6.68>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.69,   2.89,  -7.15>, <  0.13,   3.25,  -6.68>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.59,  -1.38,  -2.77>, <  1.01,  -1.64,  -2.73>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.43,  -1.89,  -2.69>, <  1.01,  -1.64,  -2.73>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.59,  -1.38,  -2.77>, <  0.43,  -1.57,  -3.19>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.27,  -1.76,  -3.62>, <  0.43,  -1.57,  -3.19>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  4.48,   1.79,  -1.01>, <  4.10,   1.70,  -0.75>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  3.71,   1.61,  -0.50>, <  4.10,   1.70,  -0.75>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  4.48,   1.79,  -1.01>, <  4.28,   1.68,  -1.43>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  4.07,   1.58,  -1.86>, <  4.28,   1.68,  -1.43>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.69,   2.89,  -7.15>, <  1.17,   3.03,  -7.13>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.65,   3.17,  -7.10>, <  1.17,   3.03,  -7.13>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.69,   2.89,  -7.15>, <  0.70,   2.55,  -6.79>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.71,   2.22,  -6.42>, <  0.70,   2.55,  -6.79>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.72,  -1.16,  -7.96>, < -2.97,  -1.10,  -7.54>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -3.22,  -1.03,  -7.12>, < -2.97,  -1.10,  -7.54>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.72,  -1.16,  -7.96>, < -3.02,  -1.01,  -8.31>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -3.31,  -0.86,  -8.65>, < -3.02,  -1.01,  -8.31>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.98,   2.80, -10.12>, <  2.42,   2.86,  -9.93>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.86,   2.92,  -9.73>, <  2.42,   2.86,  -9.93>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.98,   2.80, -10.12>, <  2.03,   2.31, -10.17>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.08,   1.82, -10.22>, <  2.03,   2.31, -10.17>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  4.08,   0.63,  -5.85>, <  4.37,   0.44,  -6.18>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  4.67,   0.25,  -6.52>, <  4.37,   0.44,  -6.18>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  4.08,   0.63,  -5.85>, <  4.03,   0.29,  -5.49>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  3.97,  -0.04,  -5.14>, <  4.03,   0.29,  -5.49>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.02,  -0.29,  -0.35>, <  1.57,  -0.25,  -0.18>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.12,  -0.21,   0.00>, <  1.57,  -0.25,  -0.18>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.02,  -0.29,  -0.35>, <  1.96,  -0.33,  -0.83>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.89,  -0.38,  -1.31>, <  1.96,  -0.33,  -0.83>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  4.47,  -0.81,  -3.64>, <  4.52,  -1.27,  -3.55>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  4.58,  -1.73,  -3.45>, <  4.52,  -1.27,  -3.55>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  4.47,  -0.81,  -3.64>, <  4.89,  -0.61,  -3.52>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  5.30,  -0.41,  -3.40>, <  4.89,  -0.61,  -3.52>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
// no constraints
