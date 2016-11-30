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

set object 15 rect from 0.158242, 0 to 155.285746, 10 fc rgb "#FF0000"
set object 16 rect from 0.107469, 0 to 103.554550, 10 fc rgb "#00FF00"
set object 17 rect from 0.109377, 0 to 104.159428, 10 fc rgb "#0000FF"
set object 18 rect from 0.109797, 0 to 104.773800, 10 fc rgb "#FFFF00"
set object 19 rect from 0.110480, 0 to 105.786052, 10 fc rgb "#FF00FF"
set object 20 rect from 0.111534, 0 to 108.746850, 10 fc rgb "#808080"
set object 21 rect from 0.114654, 0 to 109.271017, 10 fc rgb "#800080"
set object 22 rect from 0.115416, 0 to 109.793288, 10 fc rgb "#008080"
set object 23 rect from 0.115732, 0 to 149.713637, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.159592, 10 to 156.161282, 20 fc rgb "#FF0000"
set object 25 rect from 0.122330, 10 to 117.430751, 20 fc rgb "#00FF00"
set object 26 rect from 0.123994, 10 to 118.043242, 20 fc rgb "#0000FF"
set object 27 rect from 0.124418, 10 to 118.597788, 20 fc rgb "#FFFF00"
set object 28 rect from 0.125006, 10 to 119.474258, 20 fc rgb "#FF00FF"
set object 29 rect from 0.125930, 10 to 122.224238, 20 fc rgb "#808080"
set object 30 rect from 0.128823, 10 to 122.672434, 20 fc rgb "#800080"
set object 31 rect from 0.129527, 10 to 123.117786, 20 fc rgb "#008080"
set object 32 rect from 0.129764, 10 to 151.101920, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.160938, 20 to 159.191387, 30 fc rgb "#FF0000"
set object 34 rect from 0.137588, 20 to 131.946992, 30 fc rgb "#00FF00"
set object 35 rect from 0.139285, 20 to 132.515788, 30 fc rgb "#0000FF"
set object 36 rect from 0.139661, 20 to 134.367465, 30 fc rgb "#FFFF00"
set object 37 rect from 0.141648, 20 to 135.698781, 30 fc rgb "#FF00FF"
set object 38 rect from 0.143020, 20 to 138.461100, 30 fc rgb "#808080"
set object 39 rect from 0.145932, 20 to 138.992865, 30 fc rgb "#800080"
set object 40 rect from 0.146714, 20 to 139.572117, 30 fc rgb "#008080"
set object 41 rect from 0.147096, 20 to 152.401895, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.153577, 30 to 156.164141, 40 fc rgb "#FF0000"
set object 43 rect from 0.045214, 30 to 46.428754, 40 fc rgb "#00FF00"
set object 44 rect from 0.049328, 30 to 47.894919, 40 fc rgb "#0000FF"
set object 45 rect from 0.050566, 30 to 51.307705, 40 fc rgb "#FFFF00"
set object 46 rect from 0.054201, 30 to 53.917149, 40 fc rgb "#FF00FF"
set object 47 rect from 0.056902, 30 to 56.948202, 40 fc rgb "#808080"
set object 48 rect from 0.060098, 30 to 57.586331, 40 fc rgb "#800080"
set object 49 rect from 0.061195, 30 to 58.852120, 40 fc rgb "#008080"
set object 50 rect from 0.062101, 30 to 144.968585, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.155216, 40 to 153.941186, 50 fc rgb "#FF0000"
set object 52 rect from 0.071299, 40 to 69.326939, 50 fc rgb "#00FF00"
set object 53 rect from 0.073346, 40 to 70.083743, 50 fc rgb "#0000FF"
set object 54 rect from 0.073912, 40 to 71.935434, 50 fc rgb "#FFFF00"
set object 55 rect from 0.075898, 40 to 73.260100, 50 fc rgb "#FF00FF"
set object 56 rect from 0.077285, 40 to 76.043318, 50 fc rgb "#808080"
set object 57 rect from 0.080192, 40 to 76.568433, 50 fc rgb "#800080"
set object 58 rect from 0.080983, 40 to 77.194210, 50 fc rgb "#008080"
set object 59 rect from 0.081419, 40 to 146.815508, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.156706, 50 to 154.907861, 60 fc rgb "#FF0000"
set object 61 rect from 0.089803, 50 to 86.717537, 60 fc rgb "#00FF00"
set object 62 rect from 0.091659, 50 to 87.278748, 60 fc rgb "#0000FF"
set object 63 rect from 0.092028, 50 to 88.952858, 60 fc rgb "#FFFF00"
set object 64 rect from 0.093817, 50 to 90.247131, 60 fc rgb "#FF00FF"
set object 65 rect from 0.095162, 50 to 92.983824, 60 fc rgb "#808080"
set object 66 rect from 0.098052, 50 to 93.523188, 60 fc rgb "#800080"
set object 67 rect from 0.098894, 50 to 94.107180, 60 fc rgb "#008080"
set object 68 rect from 0.099213, 50 to 148.331041, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
