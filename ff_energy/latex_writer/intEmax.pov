#include "colors.inc"
#include "finish.inc"

global_settings {assumed_gamma 1 max_trace_level 6}
background {color White transmit 1.0}
camera {orthographic angle 0
  right -21.93*x up 18.31*y
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
atom(< -2.80,   4.41, -17.24>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #0
atom(< -4.30,   4.27, -18.27>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #1
atom(< -2.86,   3.09, -16.04>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #2
atom(< -2.72,   5.35, -16.66>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #3
atom(< -1.88,   4.27, -17.87>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #4
atom(< -6.11,  -3.32, -11.69>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #5
atom(< -6.59,  -4.99, -11.89>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #6
atom(< -4.58,  -3.23, -10.76>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #7
atom(< -6.89,  -2.83, -11.13>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #8
atom(< -5.95,  -2.88, -12.69>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #9
atom(< -1.03,   5.75, -13.49>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #10
atom(< -0.42,   5.87, -15.12>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #11
atom(< -1.52,   4.09, -13.15>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #12
atom(< -1.86,   6.44, -13.30>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #13
atom(< -0.12,   5.86, -12.90>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #14
atom(< -2.57,   2.56,  -7.98>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #15
atom(< -3.70,   3.62,  -8.78>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #16
atom(< -2.21,   1.29,  -9.19>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #17
atom(< -3.11,   2.18,  -7.23>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #18
atom(< -1.73,   3.26,  -7.76>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #19
atom(< -7.19,   1.68,  -9.51>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #20
atom(< -7.36,   0.10, -10.27>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #21
atom(< -7.24,   1.55,  -7.73>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #22
atom(< -8.05,   2.33,  -9.80>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #23
atom(< -6.23,   2.11,  -9.80>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #24
atom(< -7.56,  -6.80, -15.77>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #25
atom(< -7.30,  -6.93, -17.52>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #26
atom(< -7.40,  -8.31, -14.88>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #27
atom(< -8.65,  -6.47, -15.65>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #28
atom(< -6.80,  -6.12, -15.31>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #29
atom(<  3.59,  -7.52, -11.37>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #30
atom(<  1.99,  -6.70, -11.26>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #31
atom(<  4.37,  -7.03, -12.87>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #32
atom(<  3.29,  -8.57, -11.43>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #33
atom(<  4.18,  -7.31, -10.44>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #34
atom(<  2.93,   2.33, -14.30>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #35
atom(<  2.32,   1.83, -15.87>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #36
atom(<  2.57,   4.04, -14.11>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #37
atom(<  4.03,   2.24, -14.32>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #38
atom(<  2.55,   1.65, -13.49>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #39
atom(<  1.70,  -4.60, -14.48>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #40
atom(< -0.01,  -5.17, -14.34>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #41
atom(<  2.42,  -4.72, -16.14>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #42
atom(<  2.32,  -5.24, -13.78>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #43
atom(<  1.81,  -3.53, -14.23>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #44
atom(< -9.39,  -3.40, -15.94>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #45
atom(< -8.16,  -3.88, -14.74>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #46
atom(< -8.77,  -3.89, -17.54>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #47
atom(< -9.47,  -2.29, -15.90>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #48
atom(<-10.32,  -3.92, -15.71>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #49
atom(< -5.11,  -0.41, -17.48>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #50
atom(< -4.51,  -0.00, -15.84>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #51
atom(< -6.74,  -1.07, -17.33>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #52
atom(< -4.44,  -1.17, -17.96>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #53
atom(< -5.10,   0.50, -18.10>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #54
atom(<  3.82,   3.64, -10.89>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #55
atom(<  4.32,   3.54,  -9.12>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #56
atom(<  4.13,   2.03, -11.55>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #57
atom(<  4.47,   4.45, -11.39>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #58
atom(<  2.74,   3.95, -10.98>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #59
atom(< -1.94,  -0.52,  -4.39>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #60
atom(< -1.03,   0.93,  -3.74>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #61
atom(< -2.24,  -0.21,  -6.11>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #62
atom(< -2.91,  -0.61,  -3.82>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #63
atom(< -1.35,  -1.47,  -4.19>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #64
atom(< -3.44,  -3.44, -15.62>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #65
atom(< -2.09,  -2.36, -16.23>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #66
atom(< -4.65,  -4.08, -16.85>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #67
atom(< -4.01,  -2.88, -14.78>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #68
atom(< -3.07,  -4.37, -15.23>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #69
atom(<  8.27,  -5.15, -16.18>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #70
atom(< 10.04,  -4.84, -16.31>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #71
atom(<  7.83,  -6.83, -16.57>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #72
atom(<  7.68,  -4.46, -16.80>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #73
atom(<  7.95,  -4.98, -15.13>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #74
atom(< -0.78,  -5.21,  -9.65>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #75
atom(< -1.75,  -6.65,  -9.63>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #76
atom(< -0.57,  -4.68,  -7.95>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #77
atom(< -1.35,  -4.38, -10.14>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #78
atom(<  0.13,  -5.43, -10.20>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #79
atom(< -2.93,  -7.52, -17.72>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #80
atom(< -1.80,  -6.28, -18.31>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #81
atom(< -3.19,  -7.38, -15.95>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #82
atom(< -2.53,  -8.55, -17.95>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #83
atom(< -3.92,  -7.42, -18.21>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #84
atom(<  5.57,  -4.09,  -7.00>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #85
atom(<  3.90,  -4.72,  -7.15>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #86
atom(<  6.60,  -4.83,  -8.26>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #87
atom(<  5.98,  -4.43,  -6.01>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #88
atom(<  5.61,  -3.00,  -7.13>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #89
atom(<  6.06,  -3.45, -11.57>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #90
atom(<  5.55,  -3.79, -13.29>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #91
atom(<  5.73,  -1.73, -11.06>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #92
atom(<  7.12,  -3.66, -11.47>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #93
atom(<  5.45,  -4.11, -10.94>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #94
atom(<  6.05,  -0.43, -15.40>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #95
atom(<  4.63,  -1.47, -15.15>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #96
atom(<  7.52,  -1.14, -14.63>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #97
atom(<  5.82,   0.50, -14.87>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #98
atom(<  6.02,  -0.37, -16.52>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #99
cylinder {< -2.80,   4.41, -17.24>, < -2.83,   3.75, -16.64>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.86,   3.09, -16.04>, < -2.83,   3.75, -16.64>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -2.80,   4.41, -17.24>, < -3.55,   4.34, -17.76>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -4.30,   4.27, -18.27>, < -3.55,   4.34, -17.76>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -2.80,   4.41, -17.24>, < -2.34,   4.34, -17.56>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.88,   4.27, -17.87>, < -2.34,   4.34, -17.56>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.80,   4.41, -17.24>, < -2.76,   4.88, -16.95>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.72,   5.35, -16.66>, < -2.76,   4.88, -16.95>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -6.11,  -3.32, -11.69>, < -6.35,  -4.15, -11.79>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -6.59,  -4.99, -11.89>, < -6.35,  -4.15, -11.79>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -6.11,  -3.32, -11.69>, < -6.50,  -3.07, -11.41>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -6.89,  -2.83, -11.13>, < -6.50,  -3.07, -11.41>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -6.11,  -3.32, -11.69>, < -6.03,  -3.10, -12.19>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -5.95,  -2.88, -12.69>, < -6.03,  -3.10, -12.19>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -6.11,  -3.32, -11.69>, < -5.35,  -3.27, -11.23>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -4.58,  -3.23, -10.76>, < -5.35,  -3.27, -11.23>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -1.03,   5.75, -13.49>, < -1.44,   6.10, -13.39>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.86,   6.44, -13.30>, < -1.44,   6.10, -13.39>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.03,   5.75, -13.49>, < -0.72,   5.81, -14.30>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.42,   5.87, -15.12>, < -0.72,   5.81, -14.30>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -1.03,   5.75, -13.49>, < -1.28,   4.92, -13.32>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.52,   4.09, -13.15>, < -1.28,   4.92, -13.32>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -1.03,   5.75, -13.49>, < -0.58,   5.81, -13.19>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.12,   5.86, -12.90>, < -0.58,   5.81, -13.19>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.57,   2.56,  -7.98>, < -3.14,   3.09,  -8.38>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -3.70,   3.62,  -8.78>, < -3.14,   3.09,  -8.38>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -2.57,   2.56,  -7.98>, < -2.39,   1.92,  -8.58>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.21,   1.29,  -9.19>, < -2.39,   1.92,  -8.58>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -2.57,   2.56,  -7.98>, < -2.84,   2.37,  -7.61>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -3.11,   2.18,  -7.23>, < -2.84,   2.37,  -7.61>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.57,   2.56,  -7.98>, < -2.15,   2.91,  -7.87>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.73,   3.26,  -7.76>, < -2.15,   2.91,  -7.87>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -7.19,   1.68,  -9.51>, < -7.27,   0.89,  -9.89>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -7.36,   0.10, -10.27>, < -7.27,   0.89,  -9.89>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -7.19,   1.68,  -9.51>, < -7.62,   2.01,  -9.66>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -8.05,   2.33,  -9.80>, < -7.62,   2.01,  -9.66>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -7.19,   1.68,  -9.51>, < -6.71,   1.90,  -9.65>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -6.23,   2.11,  -9.80>, < -6.71,   1.90,  -9.65>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -7.19,   1.68,  -9.51>, < -7.22,   1.61,  -8.62>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -7.24,   1.55,  -7.73>, < -7.22,   1.61,  -8.62>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -7.56,  -6.80, -15.77>, < -7.43,  -6.86, -16.65>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -7.30,  -6.93, -17.52>, < -7.43,  -6.86, -16.65>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -7.56,  -6.80, -15.77>, < -8.10,  -6.63, -15.71>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -8.65,  -6.47, -15.65>, < -8.10,  -6.63, -15.71>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -7.56,  -6.80, -15.77>, < -7.18,  -6.46, -15.54>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -6.80,  -6.12, -15.31>, < -7.18,  -6.46, -15.54>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -7.56,  -6.80, -15.77>, < -7.48,  -7.55, -15.32>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -7.40,  -8.31, -14.88>, < -7.48,  -7.55, -15.32>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  3.59,  -7.52, -11.37>, <  2.79,  -7.11, -11.31>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  1.99,  -6.70, -11.26>, <  2.79,  -7.11, -11.31>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  3.59,  -7.52, -11.37>, <  3.44,  -8.05, -11.40>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  3.29,  -8.57, -11.43>, <  3.44,  -8.05, -11.40>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.59,  -7.52, -11.37>, <  3.89,  -7.42, -10.91>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  4.18,  -7.31, -10.44>, <  3.89,  -7.42, -10.91>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.59,  -7.52, -11.37>, <  3.98,  -7.28, -12.12>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  4.37,  -7.03, -12.87>, <  3.98,  -7.28, -12.12>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  2.93,   2.33, -14.30>, <  2.63,   2.08, -15.09>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  2.32,   1.83, -15.87>, <  2.63,   2.08, -15.09>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  2.93,   2.33, -14.30>, <  3.48,   2.28, -14.31>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  4.03,   2.24, -14.32>, <  3.48,   2.28, -14.31>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.93,   2.33, -14.30>, <  2.75,   3.19, -14.21>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  2.57,   4.04, -14.11>, <  2.75,   3.19, -14.21>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  2.93,   2.33, -14.30>, <  2.74,   1.99, -13.90>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  2.55,   1.65, -13.49>, <  2.74,   1.99, -13.90>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.70,  -4.60, -14.48>, <  0.84,  -4.88, -14.41>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.01,  -5.17, -14.34>, <  0.84,  -4.88, -14.41>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  1.70,  -4.60, -14.48>, <  2.06,  -4.66, -15.31>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  2.42,  -4.72, -16.14>, <  2.06,  -4.66, -15.31>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  1.70,  -4.60, -14.48>, <  2.01,  -4.92, -14.13>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  2.32,  -5.24, -13.78>, <  2.01,  -4.92, -14.13>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.70,  -4.60, -14.48>, <  1.75,  -4.06, -14.36>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  1.81,  -3.53, -14.23>, <  1.75,  -4.06, -14.36>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -9.39,  -3.40, -15.94>, < -9.08,  -3.65, -16.74>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -8.77,  -3.89, -17.54>, < -9.08,  -3.65, -16.74>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -9.39,  -3.40, -15.94>, < -9.86,  -3.66, -15.82>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<-10.32,  -3.92, -15.71>, < -9.86,  -3.66, -15.82>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -9.39,  -3.40, -15.94>, < -8.78,  -3.64, -15.34>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -8.16,  -3.88, -14.74>, < -8.78,  -3.64, -15.34>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -9.39,  -3.40, -15.94>, < -9.43,  -2.84, -15.92>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -9.47,  -2.29, -15.90>, < -9.43,  -2.84, -15.92>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -5.11,  -0.41, -17.48>, < -4.77,  -0.79, -17.72>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -4.44,  -1.17, -17.96>, < -4.77,  -0.79, -17.72>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -5.11,  -0.41, -17.48>, < -5.92,  -0.74, -17.40>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -6.74,  -1.07, -17.33>, < -5.92,  -0.74, -17.40>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -5.11,  -0.41, -17.48>, < -5.10,   0.05, -17.79>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -5.10,   0.50, -18.10>, < -5.10,   0.05, -17.79>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -5.11,  -0.41, -17.48>, < -4.81,  -0.21, -16.66>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -4.51,  -0.00, -15.84>, < -4.81,  -0.21, -16.66>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  3.82,   3.64, -10.89>, <  4.15,   4.04, -11.14>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  4.47,   4.45, -11.39>, <  4.15,   4.04, -11.14>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.82,   3.64, -10.89>, <  3.98,   2.84, -11.22>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  4.13,   2.03, -11.55>, <  3.98,   2.84, -11.22>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  3.82,   3.64, -10.89>, <  3.28,   3.80, -10.94>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  2.74,   3.95, -10.98>, <  3.28,   3.80, -10.94>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.82,   3.64, -10.89>, <  4.07,   3.59, -10.01>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  4.32,   3.54,  -9.12>, <  4.07,   3.59, -10.01>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -1.94,  -0.52,  -4.39>, < -2.09,  -0.36,  -5.25>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.24,  -0.21,  -6.11>, < -2.09,  -0.36,  -5.25>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -1.94,  -0.52,  -4.39>, < -2.42,  -0.56,  -4.11>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.91,  -0.61,  -3.82>, < -2.42,  -0.56,  -4.11>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.94,  -0.52,  -4.39>, < -1.48,   0.20,  -4.07>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.03,   0.93,  -3.74>, < -1.48,   0.20,  -4.07>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -1.94,  -0.52,  -4.39>, < -1.64,  -1.00,  -4.29>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.35,  -1.47,  -4.19>, < -1.64,  -1.00,  -4.29>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.44,  -3.44, -15.62>, < -2.76,  -2.90, -15.93>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.09,  -2.36, -16.23>, < -2.76,  -2.90, -15.93>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -3.44,  -3.44, -15.62>, < -4.04,  -3.76, -16.24>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -4.65,  -4.08, -16.85>, < -4.04,  -3.76, -16.24>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -3.44,  -3.44, -15.62>, < -3.73,  -3.16, -15.20>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -4.01,  -2.88, -14.78>, < -3.73,  -3.16, -15.20>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.44,  -3.44, -15.62>, < -3.26,  -3.90, -15.43>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -3.07,  -4.37, -15.23>, < -3.26,  -3.90, -15.43>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  8.27,  -5.15, -16.18>, <  8.05,  -5.99, -16.37>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  7.83,  -6.83, -16.57>, <  8.05,  -5.99, -16.37>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  8.27,  -5.15, -16.18>, <  7.98,  -4.81, -16.49>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  7.68,  -4.46, -16.80>, <  7.98,  -4.81, -16.49>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  8.27,  -5.15, -16.18>, <  8.11,  -5.06, -15.66>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  7.95,  -4.98, -15.13>, <  8.11,  -5.06, -15.66>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  8.27,  -5.15, -16.18>, <  9.15,  -4.99, -16.24>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< 10.04,  -4.84, -16.31>, <  9.15,  -4.99, -16.24>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -0.78,  -5.21,  -9.65>, < -1.27,  -5.93,  -9.64>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.75,  -6.65,  -9.63>, < -1.27,  -5.93,  -9.64>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -0.78,  -5.21,  -9.65>, < -0.67,  -4.94,  -8.80>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.57,  -4.68,  -7.95>, < -0.67,  -4.94,  -8.80>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -0.78,  -5.21,  -9.65>, < -1.07,  -4.79,  -9.89>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.35,  -4.38, -10.14>, < -1.07,  -4.79,  -9.89>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.78,  -5.21,  -9.65>, < -0.32,  -5.32,  -9.93>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.13,  -5.43, -10.20>, < -0.32,  -5.32,  -9.93>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.93,  -7.52, -17.72>, < -2.36,  -6.90, -18.02>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.80,  -6.28, -18.31>, < -2.36,  -6.90, -18.02>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -2.93,  -7.52, -17.72>, < -2.73,  -8.03, -17.84>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.53,  -8.55, -17.95>, < -2.73,  -8.03, -17.84>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.93,  -7.52, -17.72>, < -3.42,  -7.47, -17.96>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -3.92,  -7.42, -18.21>, < -3.42,  -7.47, -17.96>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.93,  -7.52, -17.72>, < -3.06,  -7.45, -16.84>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -3.19,  -7.38, -15.95>, < -3.06,  -7.45, -16.84>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  5.57,  -4.09,  -7.00>, <  4.73,  -4.40,  -7.07>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  3.90,  -4.72,  -7.15>, <  4.73,  -4.40,  -7.07>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  5.57,  -4.09,  -7.00>, <  5.78,  -4.26,  -6.50>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  5.98,  -4.43,  -6.01>, <  5.78,  -4.26,  -6.50>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  5.57,  -4.09,  -7.00>, <  6.08,  -4.46,  -7.63>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  6.60,  -4.83,  -8.26>, <  6.08,  -4.46,  -7.63>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  5.57,  -4.09,  -7.00>, <  5.59,  -3.54,  -7.07>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  5.61,  -3.00,  -7.13>, <  5.59,  -3.54,  -7.07>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  6.06,  -3.45, -11.57>, <  5.76,  -3.78, -11.26>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  5.45,  -4.11, -10.94>, <  5.76,  -3.78, -11.26>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  6.06,  -3.45, -11.57>, <  5.80,  -3.62, -12.43>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  5.55,  -3.79, -13.29>, <  5.80,  -3.62, -12.43>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  6.06,  -3.45, -11.57>, <  6.59,  -3.56, -11.52>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  7.12,  -3.66, -11.47>, <  6.59,  -3.56, -11.52>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  6.06,  -3.45, -11.57>, <  5.90,  -2.59, -11.32>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  5.73,  -1.73, -11.06>, <  5.90,  -2.59, -11.32>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  6.05,  -0.43, -15.40>, <  6.78,  -0.79, -15.02>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  7.52,  -1.14, -14.63>, <  6.78,  -0.79, -15.02>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  6.05,  -0.43, -15.40>, <  5.94,   0.04, -15.13>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  5.82,   0.50, -14.87>, <  5.94,   0.04, -15.13>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  6.05,  -0.43, -15.40>, <  5.34,  -0.95, -15.28>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  4.63,  -1.47, -15.15>, <  5.34,  -0.95, -15.28>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  6.05,  -0.43, -15.40>, <  6.03,  -0.40, -15.96>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  6.02,  -0.37, -16.52>, <  6.03,  -0.40, -15.96>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
// no constraints
