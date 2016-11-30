set title "Offloading (axpy_kernel) Profile on 6 Devices"
set yrange [0:72.000000]
set xlabel "execution time in ms"
set xrange [0:158.400000]
set style fill pattern 2 bo 1
set style rect fs solid 1 noborder
set border 15 lw 0.2
set xtics out nomirror
unset key
set ytics out nomirror ("dev 0(sysid:0,type:HOSTCPU)" 5,"dev 1(sysid:1,type:HOSTCPU)" 15,"dev 2(sysid:0,type:THSIM)" 25,"dev 3(sysid:1,type:THSIM)" 35,"dev 4(sysid:2,type:THSIM)" 45,"dev 5(sysid:3,type:THSIM)" 55)
set object 1 rect from 4, 65 to 17, 68 fc rgb "#FF0000"
set label "ACCU_TOTAL" at 4,63 font "Helvetica,8'"

set object 2 rect from 21, 65 to 34, 68 fc rgb "#00FF00"
set label "INIT_0" at 21,63 font "Helvetica,8'"

set object 3 rect from 38, 65 to 51, 68 fc rgb "#0000FF"
set label "INIT_0.1" at 38,63 font "Helvetica,8'"

set object 4 rect from 55, 65 to 68, 68 fc rgb "#FFFF00"
set label "INIT_1" at 55,63 font "Helvetica,8'"

set object 5 rect from 72, 65 to 85, 68 fc rgb "#00FFFF"
set label "MODELING" at 72,63 font "Helvetica,8'"

set object 6 rect from 89, 65 to 102, 68 fc rgb "#FF00FF"
set label "ACC_MAPTO" at 89,63 font "Helvetica,8'"

set object 7 rect from 106, 65 to 119, 68 fc rgb "#808080"
set label "KERN" at 106,63 font "Helvetica,8'"

set object 8 rect from 123, 65 to 136, 68 fc rgb "#800000"
set label "PRE_BAR_X" at 123,63 font "Helvetica,8'"

set object 9 rect from 140, 65 to 153, 68 fc rgb "#808000"
set label "DATA_X" at 140,63 font "Helvetica,8'"

set object 10 rect from 157, 65 to 170, 68 fc rgb "#008000"
set label "POST_BAR_X" at 157,63 font "Helvetica,8'"

set object 11 rect from 174, 65 to 187, 68 fc rgb "#800080"
set label "ACC_MAPFROM" at 174,63 font "Helvetica,8'"

set object 12 rect from 191, 65 to 204, 68 fc rgb "#008080"
set label "FINI_1" at 191,63 font "Helvetica,8'"

set object 13 rect from 208, 65 to 221, 68 fc rgb "#000080"
set label "BAR_FINI_2" at 208,63 font "Helvetica,8'"

set object 14 rect from 225, 65 to 238, 68 fc rgb "(null)"
set label "PROF_BAR" at 225,63 font "Helvetica,8'"

set object 15 rect from 0.149672, 0 to 154.806592, 10 fc rgb "#FF0000"
set object 16 rect from 0.096796, 0 to 100.342987, 10 fc rgb "#00FF00"
set object 17 rect from 0.099158, 0 to 101.043900, 10 fc rgb "#0000FF"
set object 18 rect from 0.099589, 0 to 101.814860, 10 fc rgb "#FFFF00"
set object 19 rect from 0.100379, 0 to 102.870194, 10 fc rgb "#FF00FF"
set object 20 rect from 0.101402, 0 to 103.326269, 10 fc rgb "#808080"
set object 21 rect from 0.101848, 0 to 103.896104, 10 fc rgb "#800080"
set object 22 rect from 0.102649, 0 to 104.498451, 10 fc rgb "#008080"
set object 23 rect from 0.102997, 0 to 151.336783, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.146001, 10 to 151.900503, 20 fc rgb "#FF0000"
set object 25 rect from 0.063582, 10 to 66.852682, 20 fc rgb "#00FF00"
set object 26 rect from 0.066196, 10 to 67.968983, 20 fc rgb "#0000FF"
set object 27 rect from 0.067053, 10 to 68.815079, 20 fc rgb "#FFFF00"
set object 28 rect from 0.067908, 10 to 70.078682, 20 fc rgb "#FF00FF"
set object 29 rect from 0.069151, 10 to 70.704399, 20 fc rgb "#808080"
set object 30 rect from 0.069761, 10 to 71.340288, 20 fc rgb "#800080"
set object 31 rect from 0.070670, 10 to 72.001544, 20 fc rgb "#008080"
set object 32 rect from 0.071032, 10 to 147.525690, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.143952, 20 to 153.467854, 30 fc rgb "#FF0000"
set object 34 rect from 0.040560, 20 to 44.444199, 30 fc rgb "#00FF00"
set object 35 rect from 0.044250, 20 to 46.700171, 30 fc rgb "#0000FF"
set object 36 rect from 0.046127, 20 to 49.250688, 30 fc rgb "#FFFF00"
set object 37 rect from 0.048656, 20 to 51.052644, 30 fc rgb "#FF00FF"
set object 38 rect from 0.050398, 20 to 51.931312, 30 fc rgb "#808080"
set object 39 rect from 0.051421, 20 to 52.823118, 30 fc rgb "#800080"
set object 40 rect from 0.052534, 20 to 53.801258, 30 fc rgb "#008080"
set object 41 rect from 0.053100, 20 to 145.028997, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.151312, 30 to 156.647054, 40 fc rgb "#FF0000"
set object 43 rect from 0.111243, 30 to 114.709779, 40 fc rgb "#00FF00"
set object 44 rect from 0.113277, 30 to 115.414748, 40 fc rgb "#0000FF"
set object 45 rect from 0.113737, 30 to 116.351235, 40 fc rgb "#FFFF00"
set object 46 rect from 0.114681, 30 to 117.441139, 40 fc rgb "#FF00FF"
set object 47 rect from 0.115748, 30 to 117.871845, 40 fc rgb "#808080"
set object 48 rect from 0.116174, 30 to 118.413225, 40 fc rgb "#800080"
set object 49 rect from 0.116937, 30 to 119.019629, 40 fc rgb "#008080"
set object 50 rect from 0.117292, 30 to 153.133654, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.152840, 40 to 157.973743, 50 fc rgb "#FF0000"
set object 52 rect from 0.131522, 40 to 135.310251, 50 fc rgb "#00FF00"
set object 53 rect from 0.133562, 40 to 135.888200, 50 fc rgb "#0000FF"
set object 54 rect from 0.133894, 40 to 136.742469, 50 fc rgb "#FFFF00"
set object 55 rect from 0.134733, 40 to 137.771405, 50 fc rgb "#FF00FF"
set object 56 rect from 0.135749, 40 to 138.172627, 50 fc rgb "#808080"
set object 57 rect from 0.136142, 40 to 138.708982, 50 fc rgb "#800080"
set object 58 rect from 0.136942, 40 to 139.299100, 50 fc rgb "#008080"
set object 59 rect from 0.137255, 40 to 154.760881, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.147849, 50 to 153.410943, 60 fc rgb "#FF0000"
set object 61 rect from 0.080381, 50 to 83.545555, 60 fc rgb "#00FF00"
set object 62 rect from 0.082702, 50 to 84.500326, 60 fc rgb "#0000FF"
set object 63 rect from 0.083312, 50 to 85.502866, 60 fc rgb "#FFFF00"
set object 64 rect from 0.084307, 50 to 86.648652, 60 fc rgb "#FF00FF"
set object 65 rect from 0.085417, 50 to 87.046847, 60 fc rgb "#808080"
set object 66 rect from 0.085811, 50 to 87.608509, 60 fc rgb "#800080"
set object 67 rect from 0.086618, 50 to 88.274851, 60 fc rgb "#008080"
set object 68 rect from 0.087023, 50 to 149.453577, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
