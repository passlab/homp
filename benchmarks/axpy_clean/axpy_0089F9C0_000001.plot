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

set object 15 rect from 0.163517, 0 to 153.650919, 10 fc rgb "#FF0000"
set object 16 rect from 0.089520, 0 to 84.104859, 10 fc rgb "#00FF00"
set object 17 rect from 0.092163, 0 to 84.830763, 10 fc rgb "#0000FF"
set object 18 rect from 0.092680, 0 to 85.605204, 10 fc rgb "#FFFF00"
set object 19 rect from 0.093539, 0 to 86.691274, 10 fc rgb "#FF00FF"
set object 20 rect from 0.094738, 0 to 88.073375, 10 fc rgb "#808080"
set object 21 rect from 0.096237, 0 to 88.631542, 10 fc rgb "#800080"
set object 22 rect from 0.097110, 0 to 89.185120, 10 fc rgb "#008080"
set object 23 rect from 0.097460, 0 to 149.159062, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.159421, 10 to 152.475013, 20 fc rgb "#FF0000"
set object 25 rect from 0.044846, 10 to 45.503122, 20 fc rgb "#00FF00"
set object 26 rect from 0.050145, 10 to 46.968614, 20 fc rgb "#0000FF"
set object 27 rect from 0.051399, 10 to 48.853898, 20 fc rgb "#FFFF00"
set object 28 rect from 0.053490, 10 to 50.333157, 20 fc rgb "#FF00FF"
set object 29 rect from 0.055068, 10 to 52.066273, 20 fc rgb "#808080"
set object 30 rect from 0.056977, 10 to 52.792150, 20 fc rgb "#800080"
set object 31 rect from 0.058345, 10 to 53.687593, 20 fc rgb "#008080"
set object 32 rect from 0.058729, 10 to 145.026497, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.166959, 20 to 156.742265, 30 fc rgb "#FF0000"
set object 34 rect from 0.125277, 20 to 116.516476, 30 fc rgb "#00FF00"
set object 35 rect from 0.127520, 20 to 117.156203, 30 fc rgb "#0000FF"
set object 36 rect from 0.127953, 20 to 117.972818, 30 fc rgb "#FFFF00"
set object 37 rect from 0.128844, 20 to 119.170712, 30 fc rgb "#FF00FF"
set object 38 rect from 0.130149, 20 to 120.363088, 30 fc rgb "#808080"
set object 39 rect from 0.131451, 20 to 120.939583, 30 fc rgb "#800080"
set object 40 rect from 0.132359, 20 to 121.556394, 30 fc rgb "#008080"
set object 41 rect from 0.132755, 20 to 152.568510, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.165380, 30 to 155.691048, 40 fc rgb "#FF0000"
set object 43 rect from 0.107835, 30 to 100.834023, 40 fc rgb "#00FF00"
set object 44 rect from 0.110430, 30 to 101.547963, 40 fc rgb "#0000FF"
set object 45 rect from 0.110921, 30 to 102.419588, 40 fc rgb "#FFFF00"
set object 46 rect from 0.111896, 30 to 103.819089, 40 fc rgb "#FF00FF"
set object 47 rect from 0.113423, 30 to 105.042631, 40 fc rgb "#808080"
set object 48 rect from 0.114749, 30 to 105.726361, 40 fc rgb "#800080"
set object 49 rect from 0.115774, 30 to 106.448577, 40 fc rgb "#008080"
set object 50 rect from 0.116271, 30 to 150.927934, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.168518, 40 to 158.375521, 50 fc rgb "#FF0000"
set object 52 rect from 0.143931, 40 to 133.614983, 50 fc rgb "#00FF00"
set object 53 rect from 0.146186, 40 to 134.560821, 50 fc rgb "#0000FF"
set object 54 rect from 0.146964, 40 to 135.439766, 50 fc rgb "#FFFF00"
set object 55 rect from 0.147902, 40 to 136.534085, 50 fc rgb "#FF00FF"
set object 56 rect from 0.149095, 40 to 137.698956, 50 fc rgb "#808080"
set object 57 rect from 0.150370, 40 to 138.275451, 50 fc rgb "#800080"
set object 58 rect from 0.151275, 40 to 138.876692, 50 fc rgb "#008080"
set object 59 rect from 0.151656, 40 to 153.982678, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.161402, 50 to 152.692243, 60 fc rgb "#FF0000"
set object 61 rect from 0.070315, 50 to 66.538951, 60 fc rgb "#00FF00"
set object 62 rect from 0.072978, 50 to 67.394980, 60 fc rgb "#0000FF"
set object 63 rect from 0.073659, 50 to 68.436145, 60 fc rgb "#FFFF00"
set object 64 rect from 0.074829, 50 to 70.087702, 60 fc rgb "#FF00FF"
set object 65 rect from 0.076625, 50 to 71.351587, 60 fc rgb "#808080"
set object 66 rect from 0.078022, 50 to 72.088444, 60 fc rgb "#800080"
set object 67 rect from 0.079117, 50 to 73.409143, 60 fc rgb "#008080"
set object 68 rect from 0.080249, 50 to 147.260967, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
