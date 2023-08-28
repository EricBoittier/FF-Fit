#include "colors.inc"
#include "finish.inc"

global_settings {assumed_gamma 1 max_trace_level 6}
background {color White transmit 1.0}
camera {orthographic angle 0
  right -20.04*x up 32.19*y
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
atom(< -6.96,  -4.02, -27.30>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #0
atom(< -7.31,  -4.57, -29.00>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #1
atom(< -7.40,  -5.39, -26.27>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #2
atom(< -7.56,  -3.13, -27.06>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #3
atom(< -5.86,  -3.84, -27.25>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #4
atom(< -4.53,  14.32, -28.65>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #5
atom(< -5.16,  14.66, -26.99>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #6
atom(< -5.57,  13.22, -29.52>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #7
atom(< -4.54,  15.20, -29.34>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #8
atom(< -3.48,  14.03, -28.59>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #9
atom(< -4.10,  10.18, -30.22>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #10
atom(< -3.69,   8.73, -31.23>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #11
atom(< -2.86,  10.50, -28.92>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #12
atom(< -5.11,   9.93, -29.79>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #13
atom(< -4.15,  11.06, -30.92>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #14
atom(< -3.28,  -1.48, -34.39>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #15
atom(< -2.86,   0.29, -34.38>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #16
atom(< -2.72,  -2.18, -32.78>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #17
atom(< -2.88,  -1.95, -35.29>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #18
atom(< -4.39,  -1.53, -34.38>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #19
atom(< -3.09,   0.77, -25.72>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #20
atom(< -3.97,   2.06, -26.61>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #21
atom(< -1.43,   1.47, -25.43>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #22
atom(< -2.98,  -0.12, -26.32>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #23
atom(< -3.62,   0.41, -24.79>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #24
atom(< -2.25,   4.31, -22.20>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #25
atom(< -1.65,   2.63, -22.03>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #26
atom(< -0.79,   5.41, -22.30>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #27
atom(< -2.74,   4.60, -21.23>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #28
atom(< -2.91,   4.35, -23.07>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #29
atom(< -7.06,   3.89, -30.04>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #30
atom(< -8.48,   4.84, -29.43>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #31
atom(< -5.77,   3.78, -28.76>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #32
atom(< -6.67,   4.29, -30.99>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #33
atom(< -7.48,   2.89, -30.27>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #34
atom(< -3.82,   5.05, -34.10>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #35
atom(< -4.35,   3.64, -33.16>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #36
atom(< -2.80,   6.01, -32.99>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #37
atom(< -4.76,   5.64, -34.41>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #38
atom(< -3.22,   4.62, -34.94>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #39
atom(< -7.11,  11.97, -22.74>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #40
atom(< -8.20,  11.77, -21.28>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #41
atom(< -7.44,  13.60, -23.39>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #42
atom(< -7.41,  11.21, -23.51>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #43
atom(< -6.03,  11.83, -22.51>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #44
atom(< -1.10,   6.91, -27.67>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #45
atom(< -1.06,   5.13, -27.52>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #46
atom(< -1.74,   7.34, -29.32>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #47
atom(< -0.10,   7.38, -27.46>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #48
atom(< -1.82,   7.31, -26.95>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #49
atom(< -5.28,  -3.40, -23.89>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #50
atom(< -5.07,  -2.14, -25.19>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #51
atom(< -5.39,  -2.69, -22.19>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #52
atom(< -4.30,  -3.92, -23.88>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #53
atom(< -6.18,  -4.04, -24.09>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #54
atom(< -7.83,   7.07, -33.33>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #55
atom(< -9.14,   5.88, -32.93>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #56
atom(< -6.41,   6.71, -32.26>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #57
atom(< -8.27,   8.02, -33.04>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #58
atom(< -7.52,   7.01, -34.40>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #59
atom(< -0.86,   3.88, -30.87>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #60
atom(<  0.64,   3.71, -31.85>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #61
atom(< -1.08,   2.36, -29.92>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #62
atom(< -1.65,   4.05, -31.60>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #63
atom(< -0.72,   4.75, -30.20>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #64
atom(< -7.20,  11.05, -33.41>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #65
atom(< -8.21,  10.57, -32.03>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #66
atom(< -5.99,   9.82, -33.91>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #67
atom(< -6.73,  12.06, -33.25>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #68
atom(< -7.95,  11.17, -34.20>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #69
atom(< -7.45,   1.83, -21.38>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #70
atom(< -6.48,   0.92, -20.15>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #71
atom(< -6.68,   1.80, -23.02>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #72
atom(< -7.51,   2.92, -21.14>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #73
atom(< -8.45,   1.41, -21.41>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #74
atom(< -6.38,  -1.20, -30.07>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #75
atom(< -8.08,  -0.78, -30.17>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #76
atom(< -5.70,  -1.19, -28.41>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #77
atom(< -5.82,  -0.49, -30.74>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #78
atom(< -6.22,  -2.22, -30.49>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #79
atom(< -6.57,   8.14, -20.23>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #80
atom(< -6.31,   6.51, -20.96>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #81
atom(< -5.52,   9.39, -21.02>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #82
atom(< -6.33,   8.17, -19.14>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #83
atom(< -7.63,   8.44, -20.35>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #84
atom(< -2.70,  11.69, -25.32>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #85
atom(< -3.58,  12.24, -23.77>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #86
atom(< -1.42,  10.44, -25.00>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #87
atom(< -2.21,  12.62, -25.79>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #88
atom(< -3.51,  11.30, -25.98>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #89
atom(< -6.10,   6.42, -26.33>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #90
atom(< -5.41,   8.02, -26.70>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #91
atom(< -4.63,   5.66, -25.53>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #92
atom(< -6.33,   5.93, -27.27>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #93
atom(< -6.96,   6.55, -25.61>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #94
atom(<  4.99, -13.63,  -1.01>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #95
atom(<  5.95, -12.17,  -0.98>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #96
atom(<  5.76, -14.92,   0.00>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #97
atom(<  4.99, -13.94,  -2.10>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #98
atom(<  3.94, -13.42,  -0.74>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #99
cylinder {< -6.96,  -4.02, -27.30>, < -7.14,  -4.29, -28.15>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -7.31,  -4.57, -29.00>, < -7.14,  -4.29, -28.15>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -6.96,  -4.02, -27.30>, < -7.26,  -3.57, -27.18>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -7.56,  -3.13, -27.06>, < -7.26,  -3.57, -27.18>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -6.96,  -4.02, -27.30>, < -6.41,  -3.93, -27.28>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -5.86,  -3.84, -27.25>, < -6.41,  -3.93, -27.28>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -6.96,  -4.02, -27.30>, < -7.18,  -4.70, -26.78>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -7.40,  -5.39, -26.27>, < -7.18,  -4.70, -26.78>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -4.53,  14.32, -28.65>, < -4.53,  14.76, -29.00>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -4.54,  15.20, -29.34>, < -4.53,  14.76, -29.00>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.53,  14.32, -28.65>, < -4.00,  14.17, -28.62>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -3.48,  14.03, -28.59>, < -4.00,  14.17, -28.62>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.53,  14.32, -28.65>, < -5.05,  13.77, -29.09>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -5.57,  13.22, -29.52>, < -5.05,  13.77, -29.09>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -4.53,  14.32, -28.65>, < -4.85,  14.49, -27.82>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -5.16,  14.66, -26.99>, < -4.85,  14.49, -27.82>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -4.10,  10.18, -30.22>, < -3.90,   9.45, -30.72>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -3.69,   8.73, -31.23>, < -3.90,   9.45, -30.72>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -4.10,  10.18, -30.22>, < -4.60,  10.05, -30.00>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -5.11,   9.93, -29.79>, < -4.60,  10.05, -30.00>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.10,  10.18, -30.22>, < -3.48,  10.34, -29.57>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.86,  10.50, -28.92>, < -3.48,  10.34, -29.57>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -4.10,  10.18, -30.22>, < -4.13,  10.62, -30.57>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -4.15,  11.06, -30.92>, < -4.13,  10.62, -30.57>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.28,  -1.48, -34.39>, < -3.00,  -1.83, -33.58>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.72,  -2.18, -32.78>, < -3.00,  -1.83, -33.58>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -3.28,  -1.48, -34.39>, < -3.84,  -1.50, -34.38>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -4.39,  -1.53, -34.38>, < -3.84,  -1.50, -34.38>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.28,  -1.48, -34.39>, < -3.08,  -1.72, -34.84>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.88,  -1.95, -35.29>, < -3.08,  -1.72, -34.84>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.28,  -1.48, -34.39>, < -3.07,  -0.59, -34.39>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.86,   0.29, -34.38>, < -3.07,  -0.59, -34.39>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -3.09,   0.77, -25.72>, < -3.53,   1.42, -26.16>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -3.97,   2.06, -26.61>, < -3.53,   1.42, -26.16>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -3.09,   0.77, -25.72>, < -3.04,   0.33, -26.02>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.98,  -0.12, -26.32>, < -3.04,   0.33, -26.02>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.09,   0.77, -25.72>, < -3.35,   0.59, -25.25>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -3.62,   0.41, -24.79>, < -3.35,   0.59, -25.25>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.09,   0.77, -25.72>, < -2.26,   1.12, -25.57>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.43,   1.47, -25.43>, < -2.26,   1.12, -25.57>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -2.25,   4.31, -22.20>, < -1.95,   3.47, -22.11>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.65,   2.63, -22.03>, < -1.95,   3.47, -22.11>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -2.25,   4.31, -22.20>, < -2.50,   4.45, -21.72>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.74,   4.60, -21.23>, < -2.50,   4.45, -21.72>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.25,   4.31, -22.20>, < -1.52,   4.86, -22.25>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.79,   5.41, -22.30>, < -1.52,   4.86, -22.25>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -2.25,   4.31, -22.20>, < -2.58,   4.33, -22.64>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.91,   4.35, -23.07>, < -2.58,   4.33, -22.64>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -7.06,   3.89, -30.04>, < -7.27,   3.39, -30.15>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -7.48,   2.89, -30.27>, < -7.27,   3.39, -30.15>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -7.06,   3.89, -30.04>, < -6.41,   3.84, -29.40>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -5.77,   3.78, -28.76>, < -6.41,   3.84, -29.40>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -7.06,   3.89, -30.04>, < -6.86,   4.09, -30.51>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -6.67,   4.29, -30.99>, < -6.86,   4.09, -30.51>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -7.06,   3.89, -30.04>, < -7.77,   4.36, -29.73>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -8.48,   4.84, -29.43>, < -7.77,   4.36, -29.73>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -3.82,   5.05, -34.10>, < -4.09,   4.35, -33.63>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -4.35,   3.64, -33.16>, < -4.09,   4.35, -33.63>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -3.82,   5.05, -34.10>, < -3.52,   4.83, -34.52>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -3.22,   4.62, -34.94>, < -3.52,   4.83, -34.52>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.82,   5.05, -34.10>, < -4.29,   5.34, -34.26>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -4.76,   5.64, -34.41>, < -4.29,   5.34, -34.26>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.82,   5.05, -34.10>, < -3.31,   5.53, -33.55>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.80,   6.01, -32.99>, < -3.31,   5.53, -33.55>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -7.11,  11.97, -22.74>, < -7.26,  11.59, -23.13>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -7.41,  11.21, -23.51>, < -7.26,  11.59, -23.13>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -7.11,  11.97, -22.74>, < -6.57,  11.90, -22.63>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -6.03,  11.83, -22.51>, < -6.57,  11.90, -22.63>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -7.11,  11.97, -22.74>, < -7.66,  11.87, -22.01>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -8.20,  11.77, -21.28>, < -7.66,  11.87, -22.01>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -7.11,  11.97, -22.74>, < -7.28,  12.78, -23.07>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -7.44,  13.60, -23.39>, < -7.28,  12.78, -23.07>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -1.10,   6.91, -27.67>, < -1.08,   6.02, -27.59>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.06,   5.13, -27.52>, < -1.08,   6.02, -27.59>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -1.10,   6.91, -27.67>, < -1.42,   7.12, -28.49>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.74,   7.34, -29.32>, < -1.42,   7.12, -28.49>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -1.10,   6.91, -27.67>, < -0.60,   7.15, -27.56>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.10,   7.38, -27.46>, < -0.60,   7.15, -27.56>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.10,   6.91, -27.67>, < -1.46,   7.11, -27.31>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.82,   7.31, -26.95>, < -1.46,   7.11, -27.31>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -5.28,  -3.40, -23.89>, < -4.79,  -3.66, -23.89>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -4.30,  -3.92, -23.88>, < -4.79,  -3.66, -23.89>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -5.28,  -3.40, -23.89>, < -5.73,  -3.72, -23.99>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -6.18,  -4.04, -24.09>, < -5.73,  -3.72, -23.99>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -5.28,  -3.40, -23.89>, < -5.18,  -2.77, -24.54>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -5.07,  -2.14, -25.19>, < -5.18,  -2.77, -24.54>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -5.28,  -3.40, -23.89>, < -5.33,  -3.04, -23.04>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -5.39,  -2.69, -22.19>, < -5.33,  -3.04, -23.04>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -7.83,   7.07, -33.33>, < -8.48,   6.47, -33.13>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -9.14,   5.88, -32.93>, < -8.48,   6.47, -33.13>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -7.83,   7.07, -33.33>, < -7.12,   6.89, -32.79>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -6.41,   6.71, -32.26>, < -7.12,   6.89, -32.79>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -7.83,   7.07, -33.33>, < -8.05,   7.55, -33.19>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -8.27,   8.02, -33.04>, < -8.05,   7.55, -33.19>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -7.83,   7.07, -33.33>, < -7.67,   7.04, -33.87>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -7.52,   7.01, -34.40>, < -7.67,   7.04, -33.87>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.86,   3.88, -30.87>, < -0.97,   3.12, -30.40>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.08,   2.36, -29.92>, < -0.97,   3.12, -30.40>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -0.86,   3.88, -30.87>, < -0.11,   3.80, -31.36>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.64,   3.71, -31.85>, < -0.11,   3.80, -31.36>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -0.86,   3.88, -30.87>, < -0.79,   4.31, -30.54>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.72,   4.75, -30.20>, < -0.79,   4.31, -30.54>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.86,   3.88, -30.87>, < -1.26,   3.97, -31.24>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.65,   4.05, -31.60>, < -1.26,   3.97, -31.24>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -7.20,  11.05, -33.41>, < -6.60,  10.44, -33.66>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -5.99,   9.82, -33.91>, < -6.60,  10.44, -33.66>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -7.20,  11.05, -33.41>, < -6.96,  11.56, -33.33>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -6.73,  12.06, -33.25>, < -6.96,  11.56, -33.33>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -7.20,  11.05, -33.41>, < -7.57,  11.11, -33.81>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -7.95,  11.17, -34.20>, < -7.57,  11.11, -33.81>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -7.20,  11.05, -33.41>, < -7.70,  10.81, -32.72>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -8.21,  10.57, -32.03>, < -7.70,  10.81, -32.72>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -7.45,   1.83, -21.38>, < -7.06,   1.82, -22.20>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -6.68,   1.80, -23.02>, < -7.06,   1.82, -22.20>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -7.45,   1.83, -21.38>, < -6.96,   1.38, -20.76>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -6.48,   0.92, -20.15>, < -6.96,   1.38, -20.76>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -7.45,   1.83, -21.38>, < -7.95,   1.62, -21.39>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -8.45,   1.41, -21.41>, < -7.95,   1.62, -21.39>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -7.45,   1.83, -21.38>, < -7.48,   2.38, -21.26>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -7.51,   2.92, -21.14>, < -7.48,   2.38, -21.26>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -6.38,  -1.20, -30.07>, < -6.30,  -1.71, -30.28>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -6.22,  -2.22, -30.49>, < -6.30,  -1.71, -30.28>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -6.38,  -1.20, -30.07>, < -6.10,  -0.85, -30.41>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -5.82,  -0.49, -30.74>, < -6.10,  -0.85, -30.41>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -6.38,  -1.20, -30.07>, < -6.04,  -1.20, -29.24>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -5.70,  -1.19, -28.41>, < -6.04,  -1.20, -29.24>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -6.38,  -1.20, -30.07>, < -7.23,  -0.99, -30.12>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -8.08,  -0.78, -30.17>, < -7.23,  -0.99, -30.12>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -6.57,   8.14, -20.23>, < -6.44,   7.33, -20.60>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -6.31,   6.51, -20.96>, < -6.44,   7.33, -20.60>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -6.57,   8.14, -20.23>, < -6.45,   8.15, -19.68>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -6.33,   8.17, -19.14>, < -6.45,   8.15, -19.68>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -6.57,   8.14, -20.23>, < -7.10,   8.29, -20.29>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -7.63,   8.44, -20.35>, < -7.10,   8.29, -20.29>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -6.57,   8.14, -20.23>, < -6.04,   8.77, -20.62>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -5.52,   9.39, -21.02>, < -6.04,   8.77, -20.62>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -2.70,  11.69, -25.32>, < -2.06,  11.06, -25.16>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.42,  10.44, -25.00>, < -2.06,  11.06, -25.16>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -2.70,  11.69, -25.32>, < -3.10,  11.49, -25.65>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -3.51,  11.30, -25.98>, < -3.10,  11.49, -25.65>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.70,  11.69, -25.32>, < -2.46,  12.16, -25.55>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.21,  12.62, -25.79>, < -2.46,  12.16, -25.55>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.70,  11.69, -25.32>, < -3.14,  11.97, -24.54>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -3.58,  12.24, -23.77>, < -3.14,  11.97, -24.54>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -6.10,   6.42, -26.33>, < -5.75,   7.22, -26.52>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -5.41,   8.02, -26.70>, < -5.75,   7.22, -26.52>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -6.10,   6.42, -26.33>, < -5.36,   6.04, -25.93>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -4.63,   5.66, -25.53>, < -5.36,   6.04, -25.93>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -6.10,   6.42, -26.33>, < -6.21,   6.18, -26.80>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -6.33,   5.93, -27.27>, < -6.21,   6.18, -26.80>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -6.10,   6.42, -26.33>, < -6.53,   6.49, -25.97>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -6.96,   6.55, -25.61>, < -6.53,   6.49, -25.97>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  4.99, -13.63,  -1.01>, <  5.47, -12.90,  -1.00>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  5.95, -12.17,  -0.98>, <  5.47, -12.90,  -1.00>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  4.99, -13.63,  -1.01>, <  5.38, -14.27,  -0.51>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  5.76, -14.92,   0.00>, <  5.38, -14.27,  -0.51>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  4.99, -13.63,  -1.01>, <  4.99, -13.78,  -1.56>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  4.99, -13.94,  -2.10>, <  4.99, -13.78,  -1.56>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  4.99, -13.63,  -1.01>, <  4.47, -13.53,  -0.88>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  3.94, -13.42,  -0.74>, <  4.47, -13.53,  -0.88>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
// no constraints
