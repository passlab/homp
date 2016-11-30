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

set object 15 rect from 0.554541, 0 to 146.771005, 10 fc rgb "#FF0000"
set object 16 rect from 0.537674, 0 to 142.089298, 10 fc rgb "#00FF00"
set object 17 rect from 0.539227, 0 to 142.239321, 10 fc rgb "#0000FF"
set object 18 rect from 0.539622, 0 to 142.394083, 10 fc rgb "#FFFF00"
set object 19 rect from 0.540178, 0 to 142.616875, 10 fc rgb "#FF00FF"
set object 20 rect from 0.541034, 0 to 142.705201, 10 fc rgb "#808080"
set object 21 rect from 0.541360, 0 to 142.823048, 10 fc rgb "#800080"
set object 22 rect from 0.542044, 0 to 142.951447, 10 fc rgb "#008080"
set object 23 rect from 0.542295, 0 to 146.095526, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.553306, 10 to 146.520798, 20 fc rgb "#FF0000"
set object 25 rect from 0.526403, 10 to 139.128464, 20 fc rgb "#00FF00"
set object 26 rect from 0.527996, 10 to 139.287979, 20 fc rgb "#0000FF"
set object 27 rect from 0.528399, 10 to 139.432722, 20 fc rgb "#FFFF00"
set object 28 rect from 0.528970, 10 to 139.686881, 20 fc rgb "#FF00FF"
set object 29 rect from 0.529942, 10 to 139.812648, 20 fc rgb "#808080"
set object 30 rect from 0.530398, 10 to 139.943947, 20 fc rgb "#800080"
set object 31 rect from 0.531104, 10 to 140.071294, 20 fc rgb "#008080"
set object 32 rect from 0.531373, 10 to 145.756467, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.551995, 20 to 146.156422, 30 fc rgb "#FF0000"
set object 34 rect from 0.515420, 20 to 136.212716, 30 fc rgb "#00FF00"
set object 35 rect from 0.516940, 20 to 136.361938, 30 fc rgb "#0000FF"
set object 36 rect from 0.517322, 20 to 136.552302, 30 fc rgb "#FFFF00"
set object 37 rect from 0.518031, 20 to 136.790376, 30 fc rgb "#FF00FF"
set object 38 rect from 0.518925, 20 to 136.877382, 30 fc rgb "#808080"
set object 39 rect from 0.519261, 20 to 137.002882, 30 fc rgb "#800080"
set object 40 rect from 0.519965, 20 to 137.138401, 30 fc rgb "#008080"
set object 41 rect from 0.520252, 20 to 145.430593, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.550728, 30 to 145.852965, 40 fc rgb "#FF0000"
set object 43 rect from 0.502757, 30 to 132.896204, 40 fc rgb "#00FF00"
set object 44 rect from 0.504371, 30 to 133.071803, 40 fc rgb "#0000FF"
set object 45 rect from 0.504837, 30 to 133.251614, 40 fc rgb "#FFFF00"
set object 46 rect from 0.505503, 30 to 133.505247, 40 fc rgb "#FF00FF"
set object 47 rect from 0.506466, 30 to 133.593047, 40 fc rgb "#808080"
set object 48 rect from 0.506798, 30 to 133.714587, 40 fc rgb "#800080"
set object 49 rect from 0.507480, 30 to 133.852745, 40 fc rgb "#008080"
set object 50 rect from 0.507784, 30 to 145.073335, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.549417, 40 to 145.509977, 50 fc rgb "#FF0000"
set object 52 rect from 0.489864, 40 to 129.572848, 50 fc rgb "#00FF00"
set object 53 rect from 0.491781, 40 to 129.734995, 50 fc rgb "#0000FF"
set object 54 rect from 0.492169, 40 to 129.913227, 50 fc rgb "#FFFF00"
set object 55 rect from 0.492857, 40 to 130.182417, 50 fc rgb "#FF00FF"
set object 56 rect from 0.493886, 40 to 130.277595, 50 fc rgb "#808080"
set object 57 rect from 0.494232, 40 to 130.406788, 50 fc rgb "#800080"
set object 58 rect from 0.494970, 40 to 130.559703, 50 fc rgb "#008080"
set object 59 rect from 0.495320, 40 to 144.707647, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.547862, 50 to 145.698997, 60 fc rgb "#FF0000"
set object 61 rect from 0.471636, 50 to 124.981306, 60 fc rgb "#00FF00"
set object 62 rect from 0.474431, 50 to 125.236525, 60 fc rgb "#0000FF"
set object 63 rect from 0.475113, 50 to 125.703725, 60 fc rgb "#FFFF00"
set object 64 rect from 0.476902, 50 to 126.119764, 60 fc rgb "#FF00FF"
set object 65 rect from 0.478486, 50 to 126.268467, 60 fc rgb "#808080"
set object 66 rect from 0.479042, 50 to 126.439054, 60 fc rgb "#800080"
set object 67 rect from 0.480058, 50 to 126.684782, 60 fc rgb "#008080"
set object 68 rect from 0.480616, 50 to 144.249947, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
