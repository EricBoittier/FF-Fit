#include "colors.inc"
#include "finish.inc"

global_settings {assumed_gamma 1 max_trace_level 6}
background {color White transmit 1.0}
camera {orthographic angle 0
  right -19.64*x up 23.37*y
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
atom(< -5.46,  -3.78,  -9.30>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #0
atom(< -5.08,  -5.50,  -8.98>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #1
atom(< -6.45,  -3.15,  -7.92>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #2
atom(< -6.01,  -3.66, -10.24>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #3
atom(< -4.48,  -3.26,  -9.35>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #4
atom(<  0.33,   0.88,  -7.00>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #5
atom(<  0.42,  -0.72,  -7.85>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #6
atom(< -0.91,   1.84,  -7.90>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #7
atom(<  1.33,   1.35,  -7.11>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #8
atom(< -0.04,   0.77,  -5.94>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #9
atom(<  1.73,   1.48,  -2.85>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #10
atom(<  0.23,   0.40,  -2.96>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #11
atom(<  2.00,   2.16,  -4.45>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #12
atom(<  2.61,   0.83,  -2.53>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #13
atom(<  1.46,   2.32,  -2.15>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #14
atom(< -4.01,   9.70,  -9.88>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #15
atom(< -4.69,   8.09, -10.28>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #16
atom(< -2.24,   9.60,  -9.80>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #17
atom(< -4.35,  10.13,  -8.89>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #18
atom(< -4.21,  10.36, -10.72>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #19
atom(< -1.52,   7.47,  -1.86>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #20
atom(< -0.21,   6.50,  -1.20>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #21
atom(< -2.18,   8.47,  -0.58>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #22
atom(< -2.32,   6.81,  -2.25>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #23
atom(< -1.15,   8.16,  -2.65>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #24
atom(< -5.03,   1.77, -10.35>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #25
atom(< -5.45,   0.16,  -9.82>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #26
atom(< -4.88,   1.56, -12.10>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #27
atom(< -4.05,   1.99,  -9.88>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #28
atom(< -5.76,   2.57, -10.13>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #29
atom(<  5.40,  -4.33, -11.42>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #30
atom(<  5.81,  -5.86, -12.26>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #31
atom(<  3.84,  -4.55, -10.51>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #32
atom(<  5.35,  -3.59, -12.25>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #33
atom(<  6.20,  -4.08, -10.71>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #34
atom(<  3.10,   3.90, -13.72>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #35
atom(<  1.79,   4.22, -12.55>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #36
atom(<  4.18,   5.19, -13.41>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #37
atom(<  2.81,   3.97, -14.80>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #38
atom(<  3.52,   2.92, -13.51>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #39
atom(< -4.95,  -3.89,  -0.75>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #40
atom(< -3.97,  -2.95,  -1.96>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #41
atom(< -6.08,  -4.63,  -1.90>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #42
atom(< -5.43,  -3.25,   0.00>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #43
atom(< -4.36,  -4.62,  -0.28>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #44
atom(< -4.40,  -0.57,  -5.80>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #45
atom(< -5.00,  -2.10,  -5.17>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #46
atom(< -5.19,   0.61,  -4.68>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #47
atom(< -4.76,  -0.40,  -6.86>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #48
atom(< -3.28,  -0.63,  -5.66>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #49
atom(< -4.58,   4.52,  -1.47>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #50
atom(< -3.76,   4.24,  -2.91>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #51
atom(< -6.19,   3.95,  -1.88>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #52
atom(< -4.58,   5.62,  -1.25>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #53
atom(< -4.12,   3.86,  -0.74>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #54
atom(< -2.03,   6.17, -17.00>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #55
atom(< -3.56,   6.42, -16.07>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #56
atom(< -2.36,   4.68, -17.92>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #57
atom(< -1.23,   5.96, -16.25>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #58
atom(< -1.82,   7.01, -17.69>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #59
atom(< -6.85,  -8.88,  -6.43>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #60
atom(< -6.62,  -7.19,  -6.12>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #61
atom(< -8.45,  -9.21,  -5.87>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #62
atom(< -6.10,  -9.48,  -5.92>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #63
atom(< -6.78,  -9.03,  -7.53>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #64
atom(< -7.32,   0.62, -16.13>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #65
atom(< -8.94,   1.12, -15.63>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #66
atom(< -7.53,  -0.22, -17.67>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #67
atom(< -6.91,  -0.08, -15.37>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #68
atom(< -6.71,   1.55, -16.22>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #69
atom(<  1.34,  -3.54,  -2.87>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #70
atom(<  1.43,  -3.19,  -1.11>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #71
atom(<  2.95,  -3.86,  -3.61>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #72
atom(<  0.79,  -2.74,  -3.43>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #73
atom(<  0.81,  -4.53,  -2.96>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #74
atom(<  4.30,   3.16,  -9.41>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #75
atom(<  3.05,   1.93,  -9.16>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #76
atom(<  4.32,   4.35,  -8.06>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #77
atom(<  4.05,   3.78, -10.31>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #78
atom(<  5.34,   2.78,  -9.53>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #79
atom(< -1.57,   1.57, -17.50>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #80
atom(< -2.79,   0.99, -18.70>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #81
atom(< -1.09,   0.19, -16.42>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #82
atom(< -1.99,   2.32, -16.83>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #83
atom(< -0.71,   1.95, -18.06>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #84
atom(< -1.85,  10.20, -13.52>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #85
atom(< -2.57,   8.59, -13.07>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #86
atom(< -1.07,   9.93, -15.13>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #87
atom(< -1.04,  10.55, -12.77>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #88
atom(< -2.64,  11.00, -13.57>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #89
atom(< -7.10, -10.39, -13.37>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #90
atom(< -7.22,  -8.71, -12.86>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #91
atom(< -5.88, -10.63, -14.64>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #92
atom(< -8.05, -10.69, -13.82>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #93
atom(< -6.81, -11.00, -12.50>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #94
atom(<  5.89,  -0.84, -11.16>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #95
atom(<  5.32,  -0.45, -12.85>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #96
atom(<  7.33,   0.13, -10.73>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #97
atom(<  6.23,  -1.94, -11.04>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #98
atom(<  5.04,  -0.76, -10.40>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #99
cylinder {< -5.46,  -3.78,  -9.30>, < -4.97,  -3.52,  -9.32>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -4.48,  -3.26,  -9.35>, < -4.97,  -3.52,  -9.32>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -5.46,  -3.78,  -9.30>, < -5.27,  -4.64,  -9.14>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -5.08,  -5.50,  -8.98>, < -5.27,  -4.64,  -9.14>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -5.46,  -3.78,  -9.30>, < -5.74,  -3.72,  -9.77>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -6.01,  -3.66, -10.24>, < -5.74,  -3.72,  -9.77>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -5.46,  -3.78,  -9.30>, < -5.95,  -3.46,  -8.61>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -6.45,  -3.15,  -7.92>, < -5.95,  -3.46,  -8.61>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  0.33,   0.88,  -7.00>, <  0.14,   0.82,  -6.47>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.04,   0.77,  -5.94>, <  0.14,   0.82,  -6.47>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.33,   0.88,  -7.00>, <  0.37,   0.08,  -7.43>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.42,  -0.72,  -7.85>, <  0.37,   0.08,  -7.43>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  0.33,   0.88,  -7.00>, < -0.29,   1.36,  -7.45>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.91,   1.84,  -7.90>, < -0.29,   1.36,  -7.45>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  0.33,   0.88,  -7.00>, <  0.83,   1.12,  -7.06>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  1.33,   1.35,  -7.11>, <  0.83,   1.12,  -7.06>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.73,   1.48,  -2.85>, <  2.17,   1.15,  -2.69>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  2.61,   0.83,  -2.53>, <  2.17,   1.15,  -2.69>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.73,   1.48,  -2.85>, <  0.98,   0.94,  -2.91>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.23,   0.40,  -2.96>, <  0.98,   0.94,  -2.91>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  1.73,   1.48,  -2.85>, <  1.86,   1.82,  -3.65>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  2.00,   2.16,  -4.45>, <  1.86,   1.82,  -3.65>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  1.73,   1.48,  -2.85>, <  1.60,   1.90,  -2.50>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  1.46,   2.32,  -2.15>, <  1.60,   1.90,  -2.50>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.01,   9.70,  -9.88>, < -4.11,  10.03, -10.30>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -4.21,  10.36, -10.72>, < -4.11,  10.03, -10.30>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.01,   9.70,  -9.88>, < -3.13,   9.65,  -9.84>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.24,   9.60,  -9.80>, < -3.13,   9.65,  -9.84>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -4.01,   9.70,  -9.88>, < -4.18,   9.92,  -9.39>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -4.35,  10.13,  -8.89>, < -4.18,   9.92,  -9.39>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.01,   9.70,  -9.88>, < -4.35,   8.89, -10.08>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -4.69,   8.09, -10.28>, < -4.35,   8.89, -10.08>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -1.52,   7.47,  -1.86>, < -1.33,   7.82,  -2.25>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.15,   8.16,  -2.65>, < -1.33,   7.82,  -2.25>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.52,   7.47,  -1.86>, < -0.86,   6.98,  -1.53>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.21,   6.50,  -1.20>, < -0.86,   6.98,  -1.53>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -1.52,   7.47,  -1.86>, < -1.85,   7.97,  -1.22>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.18,   8.47,  -0.58>, < -1.85,   7.97,  -1.22>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -1.52,   7.47,  -1.86>, < -1.92,   7.14,  -2.05>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.32,   6.81,  -2.25>, < -1.92,   7.14,  -2.05>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -5.03,   1.77, -10.35>, < -5.24,   0.96, -10.08>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -5.45,   0.16,  -9.82>, < -5.24,   0.96, -10.08>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -5.03,   1.77, -10.35>, < -4.95,   1.67, -11.23>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -4.88,   1.56, -12.10>, < -4.95,   1.67, -11.23>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -5.03,   1.77, -10.35>, < -4.54,   1.88, -10.11>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -4.05,   1.99,  -9.88>, < -4.54,   1.88, -10.11>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -5.03,   1.77, -10.35>, < -5.39,   2.17, -10.24>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -5.76,   2.57, -10.13>, < -5.39,   2.17, -10.24>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  5.40,  -4.33, -11.42>, <  5.60,  -5.10, -11.84>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  5.81,  -5.86, -12.26>, <  5.60,  -5.10, -11.84>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  5.40,  -4.33, -11.42>, <  5.80,  -4.21, -11.07>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  6.20,  -4.08, -10.71>, <  5.80,  -4.21, -11.07>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  5.40,  -4.33, -11.42>, <  4.62,  -4.44, -10.97>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  3.84,  -4.55, -10.51>, <  4.62,  -4.44, -10.97>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  5.40,  -4.33, -11.42>, <  5.37,  -3.96, -11.84>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  5.35,  -3.59, -12.25>, <  5.37,  -3.96, -11.84>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.10,   3.90, -13.72>, <  2.45,   4.06, -13.13>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  1.79,   4.22, -12.55>, <  2.45,   4.06, -13.13>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  3.10,   3.90, -13.72>, <  3.64,   4.55, -13.56>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  4.18,   5.19, -13.41>, <  3.64,   4.55, -13.56>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  3.10,   3.90, -13.72>, <  2.95,   3.94, -14.26>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  2.81,   3.97, -14.80>, <  2.95,   3.94, -14.26>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.10,   3.90, -13.72>, <  3.31,   3.41, -13.62>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  3.52,   2.92, -13.51>, <  3.31,   3.41, -13.62>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.95,  -3.89,  -0.75>, < -4.46,  -3.42,  -1.36>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -3.97,  -2.95,  -1.96>, < -4.46,  -3.42,  -1.36>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -4.95,  -3.89,  -0.75>, < -4.66,  -4.26,  -0.52>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -4.36,  -4.62,  -0.28>, < -4.66,  -4.26,  -0.52>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.95,  -3.89,  -0.75>, < -5.19,  -3.57,  -0.38>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -5.43,  -3.25,   0.00>, < -5.19,  -3.57,  -0.38>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.95,  -3.89,  -0.75>, < -5.52,  -4.26,  -1.33>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -6.08,  -4.63,  -1.90>, < -5.52,  -4.26,  -1.33>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -4.40,  -0.57,  -5.80>, < -3.84,  -0.60,  -5.73>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -3.28,  -0.63,  -5.66>, < -3.84,  -0.60,  -5.73>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.40,  -0.57,  -5.80>, < -4.58,  -0.49,  -6.33>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -4.76,  -0.40,  -6.86>, < -4.58,  -0.49,  -6.33>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.40,  -0.57,  -5.80>, < -4.70,  -1.34,  -5.49>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -5.00,  -2.10,  -5.17>, < -4.70,  -1.34,  -5.49>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -4.40,  -0.57,  -5.80>, < -4.79,   0.02,  -5.24>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -5.19,   0.61,  -4.68>, < -4.79,   0.02,  -5.24>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -4.58,   4.52,  -1.47>, < -4.17,   4.38,  -2.19>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -3.76,   4.24,  -2.91>, < -4.17,   4.38,  -2.19>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -4.58,   4.52,  -1.47>, < -4.58,   5.07,  -1.36>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -4.58,   5.62,  -1.25>, < -4.58,   5.07,  -1.36>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.58,   4.52,  -1.47>, < -5.38,   4.24,  -1.67>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -6.19,   3.95,  -1.88>, < -5.38,   4.24,  -1.67>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -4.58,   4.52,  -1.47>, < -4.35,   4.19,  -1.10>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -4.12,   3.86,  -0.74>, < -4.35,   4.19,  -1.10>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.03,   6.17, -17.00>, < -1.63,   6.07, -16.63>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.23,   5.96, -16.25>, < -1.63,   6.07, -16.63>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.03,   6.17, -17.00>, < -1.93,   6.59, -17.35>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.82,   7.01, -17.69>, < -1.93,   6.59, -17.35>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.03,   6.17, -17.00>, < -2.20,   5.43, -17.46>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.36,   4.68, -17.92>, < -2.20,   5.43, -17.46>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -2.03,   6.17, -17.00>, < -2.79,   6.30, -16.53>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -3.56,   6.42, -16.07>, < -2.79,   6.30, -16.53>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -6.85,  -8.88,  -6.43>, < -6.82,  -8.95,  -6.98>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -6.78,  -9.03,  -7.53>, < -6.82,  -8.95,  -6.98>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -6.85,  -8.88,  -6.43>, < -6.47,  -9.18,  -6.17>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -6.10,  -9.48,  -5.92>, < -6.47,  -9.18,  -6.17>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -6.85,  -8.88,  -6.43>, < -7.65,  -9.04,  -6.15>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -8.45,  -9.21,  -5.87>, < -7.65,  -9.04,  -6.15>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -6.85,  -8.88,  -6.43>, < -6.74,  -8.04,  -6.27>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -6.62,  -7.19,  -6.12>, < -6.74,  -8.04,  -6.27>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -7.32,   0.62, -16.13>, < -7.42,   0.20, -16.90>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -7.53,  -0.22, -17.67>, < -7.42,   0.20, -16.90>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -7.32,   0.62, -16.13>, < -7.11,   0.27, -15.75>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -6.91,  -0.08, -15.37>, < -7.11,   0.27, -15.75>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -7.32,   0.62, -16.13>, < -7.01,   1.09, -16.18>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -6.71,   1.55, -16.22>, < -7.01,   1.09, -16.18>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -7.32,   0.62, -16.13>, < -8.13,   0.87, -15.88>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -8.94,   1.12, -15.63>, < -8.13,   0.87, -15.88>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  1.34,  -3.54,  -2.87>, <  1.08,  -4.03,  -2.92>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.81,  -4.53,  -2.96>, <  1.08,  -4.03,  -2.92>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.34,  -3.54,  -2.87>, <  1.07,  -3.14,  -3.15>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.79,  -2.74,  -3.43>, <  1.07,  -3.14,  -3.15>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.34,  -3.54,  -2.87>, <  2.14,  -3.70,  -3.24>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  2.95,  -3.86,  -3.61>, <  2.14,  -3.70,  -3.24>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  1.34,  -3.54,  -2.87>, <  1.38,  -3.36,  -1.99>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  1.43,  -3.19,  -1.11>, <  1.38,  -3.36,  -1.99>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  4.30,   3.16,  -9.41>, <  4.18,   3.47,  -9.86>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  4.05,   3.78, -10.31>, <  4.18,   3.47,  -9.86>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  4.30,   3.16,  -9.41>, <  4.31,   3.76,  -8.74>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  4.32,   4.35,  -8.06>, <  4.31,   3.76,  -8.74>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  4.30,   3.16,  -9.41>, <  3.68,   2.54,  -9.28>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  3.05,   1.93,  -9.16>, <  3.68,   2.54,  -9.28>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  4.30,   3.16,  -9.41>, <  4.82,   2.97,  -9.47>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  5.34,   2.78,  -9.53>, <  4.82,   2.97,  -9.47>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.57,   1.57, -17.50>, < -1.33,   0.88, -16.96>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.09,   0.19, -16.42>, < -1.33,   0.88, -16.96>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -1.57,   1.57, -17.50>, < -1.14,   1.76, -17.78>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.71,   1.95, -18.06>, < -1.14,   1.76, -17.78>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.57,   1.57, -17.50>, < -1.78,   1.95, -17.17>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.99,   2.32, -16.83>, < -1.78,   1.95, -17.17>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.57,   1.57, -17.50>, < -2.18,   1.28, -18.10>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.79,   0.99, -18.70>, < -2.18,   1.28, -18.10>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -1.85,  10.20, -13.52>, < -1.44,  10.38, -13.14>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.04,  10.55, -12.77>, < -1.44,  10.38, -13.14>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.85,  10.20, -13.52>, < -1.46,  10.07, -14.32>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.07,   9.93, -15.13>, < -1.46,  10.07, -14.32>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -1.85,  10.20, -13.52>, < -2.21,   9.40, -13.29>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.57,   8.59, -13.07>, < -2.21,   9.40, -13.29>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -1.85,  10.20, -13.52>, < -2.24,  10.60, -13.54>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.64,  11.00, -13.57>, < -2.24,  10.60, -13.54>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -7.10, -10.39, -13.37>, < -6.49, -10.51, -14.01>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -5.88, -10.63, -14.64>, < -6.49, -10.51, -14.01>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -7.10, -10.39, -13.37>, < -7.16,  -9.55, -13.12>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -7.22,  -8.71, -12.86>, < -7.16,  -9.55, -13.12>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -7.10, -10.39, -13.37>, < -7.58, -10.54, -13.60>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -8.05, -10.69, -13.82>, < -7.58, -10.54, -13.60>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -7.10, -10.39, -13.37>, < -6.95, -10.69, -12.94>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -6.81, -11.00, -12.50>, < -6.95, -10.69, -12.94>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  5.89,  -0.84, -11.16>, <  5.46,  -0.80, -10.78>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  5.04,  -0.76, -10.40>, <  5.46,  -0.80, -10.78>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  5.89,  -0.84, -11.16>, <  5.60,  -0.65, -12.01>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  5.32,  -0.45, -12.85>, <  5.60,  -0.65, -12.01>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  5.89,  -0.84, -11.16>, <  6.61,  -0.36, -10.94>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  7.33,   0.13, -10.73>, <  6.61,  -0.36, -10.94>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  5.89,  -0.84, -11.16>, <  6.06,  -1.39, -11.10>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  6.23,  -1.94, -11.04>, <  6.06,  -1.39, -11.10>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
// no constraints
