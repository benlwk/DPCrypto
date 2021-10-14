#include <stdint.h>
#include "../include/params.h"

uint16_t A[SABER_L*SABER_L*SABER_N] = {3341, 3231, 1388, 4951, 360, 4825, 7514, 8161, 2107, 6334, 32, 465, 6645, 828, 7384, 7822, 1250, 5976, 373, 7743, 831, 8007, 4405, 7978, 8106, 6243, 2643, 4960, 3552, 7812, 2828, 7331, 5062, 6752, 8156, 8191, 2384, 7666, 3181, 294, 7036, 4079, 4602, 5922, 7836, 4475, 882, 2542, 7588, 3610, 5108, 6430, 1537, 8055, 5218, 5022, 5568, 99, 4697, 2988, 5788, 3418, 4812, 5586, 5959, 658, 4698, 7611, 6458, 2099, 6940, 2131, 5603, 4725, 4628, 4219, 1809, 128, 3934, 958, 3071, 56, 4282, 1269, 6044, 7068, 4473, 6498, 1037, 4339, 5659, 3586, 5709, 7667, 5889, 1006, 5438, 875, 1237, 2781, 3421, 242, 660, 923, 3474, 7482, 7236, 7944, 3134, 1004, 7107, 7235, 6837, 5097, 4694, 5584, 1390, 2830, 3478, 4136, 2451, 4118, 6058, 1011, 4797, 2488, 7417, 2832, 5670, 4077, 4384, 5686, 3167, 999, 6844, 6987, 3445, 5346, 5596, 3845, 1538, 5754, 4616, 4058, 4230, 5499, 6155, 3545, 2304, 6766, 5812, 4662, 2183, 2229, 1757, 2622, 4581, 5802, 1652, 4334, 6491, 7307, 1492, 8096, 3592, 717, 6632, 2065, 3085, 4104, 5175, 1699, 773, 2473, 6351, 6874, 747, 5614, 1465, 1239, 2124, 7345, 8124, 1404, 4240, 969, 3199, 1765, 6890, 6499, 4442, 4630, 7102, 4364, 1727, 3649, 5368, 1271, 3386, 4916, 2071, 3409, 7248, 5480, 3693, 8043, 6934, 6581, 7119, 6523, 4825, 506, 397, 5273, 6230, 6930, 2767, 7268, 4395, 2933, 7989, 2634, 7725, 606, 845, 1539, 3582, 1849, 830, 6144, 2171, 782, 8153, 5244, 6800, 4158, 5991, 3290, 1239, 516, 26, 6810, 6613, 3871, 4590, 7283, 3390, 985, 7496, 1649, 781, 4976, 1783, 6161, 762, 2275, 7784, 3620, 4911, 7047, 3173, 3750, 7562, 7794, 4269, 5667, 6303, 4670, 4523, 6925, 3154, 8035, 6085, 6380, 1153, 5975, 2189, 7959, 1478, 1857, 483, 6286, 3336, 7094, 1297, 4224, 3517, 6418, 980, 6628, 709, 2023, 3513, 4877, 6323, 4511, 7996, 1399, 1124, 2574, 7551, 2912, 2505, 7744, 3712, 5321, 515, 598, 3001, 4595, 2870, 3274, 6360, 5821, 1455, 5938, 7851, 5542, 3865, 4331, 3815, 4399, 7711, 6897, 7448, 5303, 6993, 6155, 1180, 3231, 4218, 4003, 2790, 889, 2226, 1037, 7192, 3727, 615, 887, 2210, 2643, 3644, 2888, 8108, 6938, 6592, 8123, 613, 3261, 2785, 3196, 6870, 3278, 166, 5396, 1543, 1017, 909, 7667, 2129, 2072, 2621, 6982, 389, 203, 1023, 7042, 6907, 558, 749, 5620, 1084, 7656, 3017, 3000, 2914, 553, 3724, 1616, 910, 5448, 337, 5495, 3381, 3541, 5561, 4210, 4775, 7750, 7986, 5519, 6288, 2326, 5255, 46, 3836, 5480, 8116, 8038, 3176, 434, 236, 2541, 824, 2138, 1999, 2650, 6601, 1398, 6417, 6538, 919, 337, 4193, 504, 4089, 4408, 1993, 3986, 57, 695, 2461, 7629, 7711, 4906, 1031, 6883, 3845, 2133, 3362, 2597, 6157, 6705, 6615, 7556, 2037, 1470, 6731, 6190, 5201, 2535, 4777, 3417, 2954, 6497, 5944, 3126, 1000, 2977, 7014, 7927, 3624, 8184, 3774, 488, 7543, 362, 1988, 5185, 6898, 2747, 6962, 3606, 1443, 7700, 2168, 1663, 5979, 7812, 6087, 2564, 4576, 3682, 2855, 55, 74, 6914, 6587, 733, 7868, 707, 7358, 769, 7710, 2446, 7620, 7604, 5760, 4761, 4434, 6826, 1247, 5092, 7491, 3259, 337, 1156, 5061, 5067, 5941, 7109, 5777, 1968, 2415, 6115, 2584, 4740, 1803, 5731, 4934, 6998, 3923, 5689, 483, 859, 701, 7558, 1161, 5663, 6318, 364, 412, 2778, 6344, 7853, 4070, 1705, 2534, 4508, 5696, 7780, 5516, 6349, 1099, 3167, 1359, 286, 6517, 4630, 81, 6527, 2604, 7863, 412, 5791, 2014, 7944, 8098, 4518, 3699, 927, 3840, 5176, 129, 1589, 7788, 6885, 2655, 2562, 6170, 6147, 1045, 3401, 1184, 5850, 72, 2282, 6370, 6548, 2033, 2016, 4178, 4275, 2797, 8054, 2841, 4799, 5963, 2013, 2050, 1405, 522, 1798, 3579, 6762, 7340, 3662, 6123, 3719, 3653, 1227, 5585, 2080, 4326, 5394, 4190, 4888, 2078, 7459, 5832, 4788, 6992, 6505, 6059, 5602, 4706, 7449, 2615, 6851, 4458, 1077, 1795, 3977, 7788, 3688, 3889, 3062, 4224, 1520, 902, 4532, 2344, 3352, 2328, 7373, 1651, 6365, 1505, 6179, 1264, 5924, 1382, 5143, 5458, 4356, 1299, 2995, 6151, 6308, 4420, 3153, 1607, 1101, 5870, 5188, 5368, 2560, 4494, 547, 367, 4473, 6793, 2763, 5622, 3214, 5653, 7669, 569, 2477, 4550, 1258, 7322, 216, 1274, 5923, 5313, 1297, 4045, 6884, 6672, 7377, 7186, 7663, 7110, 502, 2336, 6020, 6022, 7023, 7429, 1024, 1474, 2929, 6719, 7177, 5058, 7480, 6895, 7806, 5340, 4096, 7156, 2558, 1503, 49, 1013, 2748, 2546, 536, 6750, 6851, 1877, 4324, 5273, 6821, 110, 6064, 2595, 712, 5087, 5992, 4159, 2605, 7789, 2436, 2510, 573, 6010, 4317, 5536, 2912, 2685, 6811, 3273, 7627, 4949, 676, 3607, 919, 8133, 6622, 6859, 2510, 5325, 6163, 3043, 6815, 6745, 4502, 1880, 1973, 6066, 5271, 2075, 6844, 3055, 2525, 6728, 687, 5623, 6418, 5932, 3179, 2877, 7255, 7285, 3124, 6071, 6741, 4146, 6825, 3285, 4758, 6509, 7819, 3600, 1932, 7164, 5777, 3994, 1506, 479, 5292, 5070, 2163, 5461, 5365, 4535, 3547, 5795, 7140, 2618, 3809, 966, 1415, 7502, 7770, 2012, 7036, 1346, 6880, 445, 5017, 7882, 2742, 1929, 4821, 6051, 434, 4290, 1447, 881, 7442, 1483, 4133, 7023, 2276, 5179, 6691, 1856, 2861, 1251, 8140, 971, 645, 7845, 1491, 2330, 3528, 2761, 7062, 6036, 5460, 4121, 569, 1570, 4934, 1619, 1099, 713, 7555, 7901, 7391, 7693, 4528, 618, 6837, 2112, 6226, 6118, 7822, 918, 4255, 3264, 3542, 2158, 5449, 2357, 7572, 3550, 3273, 3243, 1210, 2231, 1681, 1302, 2572, 763, 7383, 4744, 1947, 4478, 5263, 4847, 7637, 2761, 3951, 7337, 3408, 123, 2612, 2664, 6167, 6183, 4237, 7682, 991, 5474, 5773, 5730, 2779, 4007, 1468, 7870, 7928, 1360, 1318, 5410, 282, 1760, 2638, 3259, 5438, 5782, 2011, 2388, 391, 5602, 3701, 3211, 7199, 5171, 2723, 3230, 3145, 6678, 2548, 4605, 6813, 6408, 5445, 4079, 3940, 5180, 7306, 1495, 5897, 1268, 5453, 6799, 927, 664, 3921, 3341, 842, 7464, 2118, 6637, 2034, 3093, 4115, 7555, 4615, 1956, 7772, 5973, 2151, 760, 6229, 6559, 3930, 1397, 1218, 4821, 106, 94, 938, 5717, 7296, 6061, 7309, 1819, 360, 7356, 2524, 5212, 6849, 3746, 2283, 3498, 8106, 4814, 4412, 493, 5556, 4673, 40, 7358, 3535, 4106, 6060, 1949, 8089, 2111, 1182, 7322, 8042, 3211, 6479, 2364, 4796, 3632, 3471, 4206, 1995, 7728, 982, 3904, 3709, 2298, 5507, 2122, 7514, 6019, 576, 6042, 5437, 7049, 5426, 1996, 1142, 2870, 1132, 2162, 3081, 1222, 4534, 4037, 2669, 7741, 1786, 7741, 4393, 6862, 7890, 602, 6421, 202, 5870, 729, 319, 7852, 6870, 7252, 2893, 119, 5716, 6063, 2687, 6686, 5912, 297, 1433, 4181, 7090, 7434, 3654, 1165, 1580, 948, 3413, 3791, 1633, 7974, 4812, 94, 740, 5403, 7871, 3456, 4230, 325, 4783, 6668, 4876, 2667, 6925, 3302, 7761, 7622, 3790, 3867, 5734, 7917, 3609, 2616, 1429, 1447, 5789, 3791, 2966, 3626, 2707, 2261, 5264, 1574, 4303, 7080, 3634, 715, 1461, 163, 7662, 4669, 6636, 1498, 7548, 6105, 5027, 3926, 7907, 3665, 6808, 4182, 4671, 1402, 2142, 7802, 4233, 144, 6603, 7854, 3899, 2055, 2374, 1844, 6343, 3075, 2775, 1602, 6284, 344, 1126, 3231, 5593, 208, 2605, 7765, 4164, 5523, 1609, 6259, 937, 3451, 252, 4455, 2785, 1116, 6258, 2709, 7516, 1964, 5495, 4376, 2137, 6973, 4913, 6151, 4250, 5895, 2669, 4480, 464, 1112, 730, 300, 4367, 1488, 2318, 4037, 5406, 1535, 3505, 2367, 1098, 6273, 4134, 7488, 1844, 3800, 2790, 2970, 8011, 4416, 5496, 891, 598, 879, 2752, 6514, 1202, 7556, 7423, 4845, 1295, 762, 4404, 2884, 2631, 781, 728, 5516, 6089, 4639, 3855, 5433, 973, 4122, 7516, 3039, 1753, 6203, 1401, 5707, 3793, 1580, 3758, 835, 4424, 2829, 2314, 7572, 503, 976, 445, 1247, 7063, 7547, 2799, 2288, 6486, 1365, 3634, 5611, 5032, 7904, 2753, 2992, 6265, 3879, 1302, 7687, 7403, 2563, 5462, 776, 4897, 686, 1495, 3041, 6043, 2471, 7665, 1958, 1848, 81, 231, 1811, 3360, 3185, 363, 4027, 2037, 750, 4472, 5879, 6617, 3877, 7145, 5162, 4027, 3234, 6034, 998, 1405, 5972, 4341, 5097, 6027, 3926, 6428, 4258, 2490, 1917, 7388, 1140, 239, 942, 1818, 59, 2803, 4502, 6320, 1812, 1105, 1080, 7383, 1043, 5485, 3542, 6891, 616, 172, 2274, 1922, 7659, 5867, 6107, 5150, 4635, 3997, 7558, 8094, 8151, 7567, 3316, 503, 7326, 3227, 6179, 2536, 326, 806, 460, 8101, 1736, 821, 355, 1307, 3136, 788, 5412, 6611, 313, 7834, 3367, 6522, 7227, 7878, 2016, 5485, 6458, 4600, 5075, 5627, 5137, 1828, 7551, 1879, 5987, 7866, 2648, 7357, 6667, 1977, 6450, 3923, 99, 7842, 5498, 6670, 4737, 7870, 5711, 6670, 2912, 1799, 261, 2454, 5493, 140, 1855, 2411, 5668, 7203, 3097, 2526, 3305, 798, 3900, 6503, 4288, 1705, 4652, 2259, 1659, 6142, 5092, 4855, 3369, 5122, 3615, 2281, 6098, 2403, 2186, 2376, 6063, 552, 6059, 2333, 1563, 2256, 3169, 5849, 7086, 7281, 1878, 3158, 7955, 2670, 226, 4743, 4189, 1966, 3345, 2712, 31, 8132, 2059, 6438, 4450, 3567, 8189, 7322, 7186, 7819, 1057, 5215, 6162, 7456, 7724, 3671, 97, 7170, 2572, 4755, 492, 4811, 1707, 3507, 3157, 246, 1426, 4104, 6090, 776, 3000, 3967, 3615, 1679, 4807, 6428, 2688, 2403, 5374, 3945, 2234, 6695, 6198, 812, 7579, 7427, 5745, 1693, 1768, 4392, 2226, 1124, 1321, 5130, 7411, 8090, 4660, 2843, 7571, 2691, 1846, 5892, 395, 237, 5955, 3163, 411, 2128, 844, 3544, 7763, 3290, 4283, 6143, 1108, 7371, 7064, 5128, 5300, 6961, 5154, 5880, 2633, 1417, 4634, 3183, 5193, 1401, 284, 4754, 1348, 7615, 6779, 6222, 7996, 6366, 675, 5910, 4228, 8084, 339, 6539, 6898, 5498, 4272, 7684, 2743, 1607, 6172, 3131, 4039, 4740, 6155, 6238, 805, 6444, 1901, 7484, 825, 423, 7557, 4104, 502, 2473, 1815, 1649, 419, 2726, 264, 5904, 6111, 25, 6159, 3728, 2962, 1531, 4503, 5780, 7176, 6136, 4229, 210, 3583, 507, 1332, 2354, 746, 4337, 5304, 142, 2451, 3658, 4024, 2996, 5660, 7202, 4108, 6668, 6449, 691, 3514, 1726, 3245, 7251, 1173, 3697, 3290, 6980, 1105, 5686, 350, 6366, 814, 3374, 1269, 3229, 4623, 1957, 1254, 2962, 4749, 264, 4017, 1521, 4445, 912, 5912, 7723, 982, 4037, 7645, 667, 8121, 5886, 4754, 3878, 3704, 2321, 7117, 1637, 2395, 2587, 3274, 4913, 3866, 1318, 4852, 6418, 4162, 5164, 4954, 4775, 2028, 778, 6838, 5737, 1602, 303, 2774, 1328, 2631, 7808, 4075, 3538, 2701, 5348, 696, 7044, 3065, 1464, 4951, 703, 2172, 7027, 3018, 2179, 4116, 6240, 2955, 3236, 6512, 5432, 4419, 8168, 5827, 1752, 7133, 5828, 5111, 674, 981, 7994, 6298, 976, 976, 5262, 1807, 7024, 4122, 3400, 2078, 3685, 2911, 7128, 250, 7548, 1045, 4656, 1122, 496, 4072, 4104, 4491, 7659, 2247, 2761, 4149, 2920, 6634, 711, 830, 3728, 3323, 7500, 7505, 6204, 6104, 2254, 7448, 4455, 6854, 1531, 3782, 3060, 2598, 6090, 3616, 4419, 2750, 3582, 4868, 4163, 496, 259, 2275, 2328, 3323, 7379, 6554, 7936, 1567, 5125, 3086, 82, 137, 4451, 804, 6295, 676, 1926, 6341, 1525, 4403, 6918, 2693, 6209, 3645, 568, 5618, 291, 2255, 3130, 6655, 599, 454, 887, 984, 2639, 8088, 3799, 4099, 2966, 7900, 4565, 2485, 6257, 6235, 4155, 313, 5747, 2771, 3617, 7143, 3954, 2178, 6002, 5778, 7506, 14, 4849, 7385, 1125, 4130, 575, 732, 8131, 7803, 1137, 4239, 4304, 391, 4945, 1201, 7352, 670, 5683, 4407, 3740, 3780, 8148, 6401, 2488, 1035, 5256, 7077, 3934, 1250, 2406, 6663, 3263, 7518, 826, 1605, 231, 6382, 2278, 2165, 1898, 5136, 3455, 4287, 2792, 3674, 59, 1769, 3924, 7374, 3737, 4209, 3441, 3955, 2786, 6457, 5618, 1498, 3800, 2041, 6817, 3871, 7408, 7924, 2376, 224, 7480, 640, 4659, 6243, 3256, 662, 7193, 357, 469, 265, 6322, 7570, 5203, 5560, 419, 1728, 167, 1733, 5176, 3849, 271, 3074, 3047, 337, 3107, 1776, 3000, 38, 3942, 4179, 259, 562, 4008, 275, 6416, 7333, 8050, 3748, 2753, 662, 443, 7256, 2431, 6431, 5502, 2024, 449, 6653, 5271, 1906, 4024, 2165, 6660, 4402, 3047, 3654, 7772, 7053, 1061, 2013, 5327, 5104, 2147, 5185, 6648, 5062, 7018, 8, 7958, 366, 2112, 4799, 3734, 2088, 5102, 1603, 3222, 5556, 2246, 4755, 341, 7129, 1577, 5937, 1305, 1881, 2973, 2958, 5881, 6289, 2608, 8164, 1659, 204, 3358, 2358, 1069, 1633, 850, 6855, 7113, 7881, 7076, 7131, 469, 5147, 2443, 5669, 5219, 1502, 4163, 7347, 2746, 7957, 1103, 3079, 3340, 1224, 337, 1902, 970, 7073, 138, 1317, 6831, 7046, 2666, 1143, 4275, 5158, 4717, 7112, 7272, 5780, 2449, 7472, 6185, 2844, 8123, 289, 2182, 4482, 1962, 2584, 4311, 6337, 3328, 3524, 4808, 6727, 5474, 6516, 8092, 4181, 7068, 3659, 2007, 7580, 5153, 2307, 3306, 480, 5995, 1250, 4065, 2441, 5127, 3214, 5980, 1137, 3593, 4089, 5928, 7060, 2156, 1612, 2790, 6712, 5210, 4589, 188, 395, 7381, 6788, 5945, 7355, 3333, 273, 7393, 1940, 154, 955, 3033, 2349, 5058, 6332, 4746, 878, 4188, 2670, 4268, 3934, 371, 4612, 6164, 1261, 4382, 4939, 2624, 3415, 6802, 4943, 4992, 7673, 5954, 1705, 1945, 7011, 4262, 3400, 6901, 8093, 1184, 4009, 4434, 3558, 2009, 4177, 4045, 1140, 2756, 5870, 7384, 6050, 7255, 5461, 5727, 2220, 3854, 4516, 4094, 789, 3572, 2679, 1358, 7187, 3491, 6565, 7059, 1956, 10, 35, 6008, 6111, 413, 6238, 5462, 2894, 4326, 1969, 7331, 3277, 2043, 5802, 7103, 4217, 2159, 1529, 7000, 7587, 6679, 3618, 2829, 6552, 2118, 3164, 1451, 97, 4560, 3505, 788, 4713, 5468, 1450, 5249, 7254, 1618, 3880, 7303, 3213, 3011, 5772, 2644, 7602, 3859, 6155, 4658, 1050, 3780, 5127, 369, 6684, 4538, 7276, 5503, 766, 533, 2782, 4745, 2886, 4616, 2033, 1755, 1129, 7836, 5630, 6635, 7288, 5510, 5370, 4352, 2743, 6803, 149, 2365, 1402, 1315, 3043, 4441, 4060, 1177, 2020, 2798, 455, 6821, 4669, 6899, 2746, 4982, 2469, 6142, 3276, 4905, 3898, 7004, 4559, 3945, 3198, 702, 4427, 5993, 4834, 7749, 6811, 3271, 3011, 1263, 231, 2773, 2202, 6432, 4942, 4458, 7601, 4742, 5535, 5517, 1491, 1703, 6641, 3391, 6318, 409, 6076, 18, 2739, 5693, 5362, 5453, 8000, 3779, 1086, 7286, 7017, 1854, 2116, 4534, 180, 6271, 3671, 2245, 7842, 5437, 4314, 5597, 5896, 339, 7872, 5265, 2025, 2710, 2784, 7168, 3843, 3618, 5461, 4078, 3978, 6371, 3806, 2403, 40, 6363, 5949, 2977, 6984, 6643, 3921, 3590, 1382, 4918, 1751, 6264, 557, 111, 2230, 2343, 531, 1448, 6660, 5647, 6966, 7908, 1377, 1926, 3340, 8164, 2001, 1232, 5616, 2920, 3492, 4802, 2384, 1794, 3399, 2699, 3608, 2520, 56, 1150, 918, 6863, 2007, 4882, 1066, 1509, 6447, 447, 5757, 3455, 6945, 6871, 2457, 5937, 6929, 4645, 2934, 5391, 787, 2666, 4091, 6178, 6677, 5587, 6139, 1850, 4364, 3880, 6566, 511, 2197, 2955};  

uint16_t sp[SABER_N*SABER_L] = {2, 65533, 65535, 1, 0, 65534, 2, 65535, 0, 0, 1, 1, 0, 65535, 1, 65534, 0, 1, 1, 65534, 65534, 0, 1, 0, 3, 65535, 65535, 65535, 65535, 0, 65532, 1, 1, 2, 65535, 65535, 65535, 2, 65533, 65535, 0, 2, 2, 65535, 0, 1, 65534, 65535, 1, 65534, 1, 1, 65535, 0, 0, 0, 1, 2, 2, 0, 0, 2, 0, 65535, 1, 1, 2, 2, 0, 65535, 2, 65535, 65535, 2, 0, 65534, 0, 65535, 65534, 3, 0, 65534, 65535, 1, 2, 65535, 2, 0, 65534, 1, 65535, 65534, 1, 0, 65535, 65535, 0, 0, 1, 2, 0, 0, 0, 0, 65535, 1, 65534, 65535, 0, 1, 2, 0, 0, 2, 65535, 3, 0, 3, 0, 0, 0, 0, 3, 0, 65533, 1, 2, 3, 2, 4, 65533, 65535, 0, 1, 65534, 1, 0, 1, 65535, 2, 2, 2, 0, 65535, 1, 65535, 0, 0, 65533, 65535, 1, 65534, 65535, 65535, 65534, 65535, 0, 65534, 2, 2, 65534, 65534, 65535, 65533, 65535, 0, 0, 65535, 1, 65535, 1, 65535, 0, 0, 1, 0, 65535, 1, 1, 2, 1, 65534, 65535, 65535, 65535, 65534, 65534, 65534, 1, 65534, 1, 1, 0, 0, 65535, 1, 0, 65535, 0, 0, 0, 0, 2, 0, 1, 65534, 0, 1, 65535, 0, 65535, 65533, 65535, 2, 1, 65533, 65534, 0, 0, 0, 2, 0, 65535, 65534, 0, 0, 65535, 1, 65534, 65534, 1, 1, 1, 65535, 0, 65534, 65535, 2, 1, 65533, 65535, 0, 65534, 65534, 65534, 65534, 0, 1, 0, 1, 65535, 0, 1, 65535, 65535, 2, 0, 0, 1, 0, 65533, 65534, 1, 65535, 65535, 65535, 65535, 0, 0, 1, 65534, 1, 2, 65535, 65535, 65534, 65535, 0, 1, 1, 65535, 0, 1, 65535, 1, 1, 65535, 1, 65535, 2, 1, 1, 65535, 1, 0, 0, 0, 1, 65534, 0, 1, 0, 1, 65534, 0, 1, 2, 65534, 65535, 65535, 1, 65535, 0, 0, 1, 0, 1, 0, 65535, 65534, 2, 65534, 3, 65535, 3, 0, 1, 1, 0, 0, 0, 0, 65534, 65534, 65535, 0, 1, 0, 1, 3, 0, 65534, 1, 1, 1, 65535, 65534, 2, 2, 65535, 65535, 65535, 0, 65535, 3, 0, 65535, 0, 2, 0, 1, 65534, 2, 65534, 1, 65535, 65535, 65534, 65535, 65534, 3, 2, 65535, 1, 0, 0, 65533, 0, 1, 0, 65534, 1, 65535, 0, 2, 65534, 0, 0, 0, 65535, 0, 0, 2, 0, 0, 0, 0, 65534, 1, 0, 0, 1, 1, 2, 65534, 0, 65535, 65535, 0, 0, 2, 65535, 1, 65535, 1, 0, 65535, 65534, 65534, 0, 0, 1, 65535, 0, 0, 1, 65534, 65535, 1, 0, 0, 65535, 65535, 65533, 1, 0, 65534, 65534, 0, 2, 2, 65534, 0, 0, 65534, 2, 65535, 1, 0, 0, 65535, 65535, 1, 65535, 65535, 1, 65534, 1, 0, 1, 1, 0, 0, 2, 2, 65535, 0, 65535, 65535, 3, 65535, 0, 65535, 65534, 65534, 2, 2, 65534, 65535, 1, 1, 0, 65535, 1, 0, 0, 2, 0, 0, 65535, 1, 65534, 65534, 0, 2, 1, 3, 0, 1, 3, 65535, 0, 1, 65533, 0, 65535, 65534, 0, 0, 65535, 1, 65535, 65534, 0, 65534, 65535, 0, 3, 65535, 1, 65535, 0, 1, 0, 65534, 65535, 1, 0, 65535, 1, 65535, 0, 65535, 0, 2, 1, 0, 1, 65535, 0, 1, 0, 65535, 65534, 65534, 0, 65534, 65535, 65535, 1, 0, 65534, 0, 1, 0, 2, 1, 1, 3, 1, 0, 0, 1, 65535, 65534, 65534, 2, 65534, 65535, 2, 0, 65534, 65535, 2, 2, 65534, 1, 65534, 3, 65534, 65534, 65534, 65534, 2, 1, 1, 65535, 3, 1, 0, 1, 3, 0, 1, 2, 65534, 2, 1, 1, 1, 1, 65534, 0, 0, 2, 0, 0, 65535, 2, 0, 65535, 65534, 0, 1, 0, 65534, 65535, 1, 65535, 0, 2, 2, 65535, 0, 1, 65535, 3, 1, 0, 0, 2, 1, 2, 2, 65535, 0, 1, 65535, 0, 0, 2, 0, 0, 65535, 2, 0, 0, 1, 0, 3, 1, 0, 65534, 65534, 65535, 0, 0, 0, 65533, 65534, 0, 65535, 2, 65534, 65535, 1, 65535, 2, 1, 1, 0, 3, 65534, 65535, 65533, 65534, 1, 3, 0, 1, 65534, 65535, 65535, 65534, 1, 0, 0, 65534, 2, 65535, 0, 1, 1, 1, 65534, 0, 0, 1, 3, 1, 0, 1, 2, 65535, 1, 65535, 2, 65535, 0, 65535, 0, 65535, 1, 0, 65535, 0, 65534, 65534, 0, 65534, 0, 1, 3, 0, 0, 1, 1, 1, 65535, 2, 65535, 0, 1, 2, 3, 2, 0, 3, 1, 65534, 1, 0, 1, 0, 2, 65535, 1, 0, 0, 0, 1, 2, 0, 65535, 0, 0, 0, 65534, 65535, 65535, 1, 2, 65535, 65535, 65534, 1, 1, 2};

uint8_t pk[SABER_INDCPA_PUBLICKEYBYTES] = {235, 174, 169, 23, 122, 182, 12, 219, 33, 229, 182, 223, 150, 159, 25, 175, 115, 7, 31, 163, 5, 11, 76, 168, 144, 126, 155, 254, 56, 90, 231, 228, 233, 241, 182, 74, 191, 59, 6, 131, 97, 157, 18, 70, 211, 167, 36, 206, 44, 147, 196, 181, 119, 98, 216, 89, 71, 190, 162, 137, 105, 66, 232, 170, 30, 123, 178, 243, 255, 186, 81, 162, 150, 217, 127, 145, 51, 195, 168, 199, 77, 253, 24, 227, 141, 229, 181, 36, 14, 65, 232, 133, 145, 104, 5, 132, 36, 198, 92, 223, 156, 209, 144, 82, 2, 248, 138, 90, 247, 114, 31, 133, 6, 65, 116, 242, 249, 39, 185, 125, 105, 129, 206, 95, 38, 31, 162, 31, 250, 190, 116, 187, 59, 121, 179, 35, 165, 72, 34, 167, 5, 166, 166, 129, 87, 194, 238, 209, 243, 122, 159, 92, 105, 161, 213, 166, 103, 193, 186, 53, 212, 191, 111, 241, 210, 31, 55, 187, 9, 14, 147, 85, 221, 238, 75, 58, 135, 168, 204, 232, 56, 161, 176, 167, 246, 167, 71, 81, 75, 72, 178, 8, 152, 211, 100, 175, 204, 161, 50, 220, 15, 82, 251, 18, 7, 60, 35, 93, 40, 31, 123, 3, 52, 27, 10, 27, 11, 166, 238, 100, 220, 214, 214, 65, 56, 61, 130, 71, 30, 136, 42, 106, 149, 159, 142, 168, 66, 117, 87, 165, 163, 240, 238, 101, 5, 240, 212, 213, 44, 153, 5, 31, 182, 254, 194, 169, 78, 115, 97, 131, 103, 38, 184, 169, 39, 244, 88, 168, 157, 24, 208, 88, 159, 158, 194, 251, 40, 25, 225, 87, 111, 151, 114, 236, 11, 98, 140, 41, 232, 125, 252, 53, 237, 162, 106, 186, 29, 213, 102, 169, 202, 109, 186, 5, 81, 156, 205, 207, 100, 9, 34, 91, 24, 118, 21, 63, 67, 202, 11, 45, 239, 150, 151, 165, 165, 58, 68, 155, 70, 225, 234, 167, 188, 63, 5, 78, 93, 168, 58, 182, 140, 8, 110, 160, 201, 169, 140, 50, 41, 101, 233, 180, 70, 109, 82, 198, 27, 131, 198, 35, 69, 238, 162, 139, 225, 76, 181, 201, 111, 69, 71, 30, 247, 239, 96, 118, 175, 227, 223, 63, 167, 168, 136, 153, 198, 46, 50, 191, 220, 191, 223, 136, 151, 100, 69, 19, 128, 140, 16, 162, 9, 191, 250, 56, 247, 234, 148, 230, 172, 15, 19, 155, 195, 184, 173, 255, 196, 175, 54, 116, 86, 48, 190, 61, 167, 80, 56, 119, 126, 151, 156, 254, 192, 196, 189, 25, 71, 14, 28, 220, 47, 41, 134, 220, 178, 66, 251, 112, 127, 86, 51, 46, 117, 59, 13, 185, 142, 104, 193, 8, 151, 135, 194, 241, 219, 179, 63, 142, 2, 28, 82, 137, 214, 68, 13, 101, 97, 39, 79, 230, 255, 20, 134, 1, 191, 201, 57, 58, 26, 80, 169, 223, 26, 17, 162, 116, 71, 128, 107, 240, 73, 18, 233, 194, 16, 60, 53, 231, 186, 142, 201, 67, 50, 180, 218, 174, 160, 215, 159, 154, 33, 173, 126, 69, 9, 87, 177, 70, 141, 59, 164, 71, 119, 230, 94, 79, 52, 180, 16, 128, 206, 144, 13, 59, 203, 70, 235, 105, 113, 162, 81, 70, 189, 91, 169, 29, 12, 172, 225, 200, 48, 59, 98, 16, 241, 188, 230, 73, 204, 253, 146, 237, 16, 97, 206, 138, 70, 143, 230, 137, 200, 229, 222, 226, 140, 45, 92, 247, 26, 136, 193, 179, 253, 66, 130, 3, 155, 151, 41, 13, 233, 35, 220, 90, 2, 224, 206, 28, 111, 110, 175, 84, 80, 75, 34, 122, 68, 24, 60, 25, 74, 121, 156, 245, 112, 251, 1, 125, 106, 82, 187, 136, 223, 248, 98, 192, 253, 179, 23, 223, 219, 91, 58, 76, 87, 154, 153, 0, 35, 71, 5, 160, 205, 135, 138, 106, 35, 241, 233, 6, 75, 203, 21, 142, 146, 215, 228, 104, 117, 95, 60, 233, 109, 44, 217, 36, 14, 147, 233, 107, 239, 198, 115, 66, 35, 111, 75, 183, 254, 84, 53, 165, 146, 240, 149, 109, 242, 112, 217, 137, 194, 28, 9, 99, 94, 198, 90, 201, 178, 251, 66, 173, 171, 214, 164, 137, 227, 168, 173, 251, 241, 59, 197, 94, 150, 126, 114, 186, 165, 205, 204, 133, 4, 245, 173, 94, 225, 128, 204, 255, 86, 143, 12, 188, 207, 51, 2, 254, 210, 144, 219, 69, 17, 173, 129, 18, 105, 207, 65, 236, 156, 174, 14, 91, 146, 177, 84, 241, 198, 213, 210, 56, 77, 246, 41, 207, 170, 0, 218, 192, 84, 126, 47, 211, 124, 254, 227, 218, 215, 213, 160, 51, 246, 55, 204, 205, 174, 83, 77, 224, 9, 230, 146, 42, 61, 135, 214, 23, 118, 102, 70, 82, 136, 92, 165, 94, 179, 154, 171, 187, 171, 236, 102, 169, 205, 113, 130, 45, 72, 95, 61, 192, 160, 246, 221, 104, 89, 37, 191, 38, 52, 228, 136, 83, 194, 44, 192, 124, 177, 26, 63, 94, 185, 61, 7, 219, 185, 132, 151, 99, 52, 1, 232, 234, 22, 7, 176, 63, 12, 119, 216, 3, 239, 126, 97, 253, 34, 31, 192, 178, 58, 133, 54, 36, 217, 47, 149, 214, 6, 94, 120, 178, 236, 71, 172, 144, 35, 193, 220, 192, 107, 39, 201, 158, 96, 73, 181, 254, 103, 246, 75, 145, 48, 231, 65, 118, 92, 24, 173, 169, 190, 247, 41, 111, 74, 235, 92, 116, 94, 25, 76, 169, 45, 187, 214, 2, 240, 244, 54, 198, 12, 241, 232, 230, 45, 1, 249, 94, 205, 159, 203, 160, 79, 111, 60, 181, 193, 191, 126, 57, 7, 136, 0, 167, 167, 102, 190, 94, 36, 74, 211, 252, 226, 158, 57, 131, 156, 254, 130, 201, 170, 171, 242, 124, 207, 47, 198, 83, 86, 45, 147, 143};
uint16_t m_tv[SABER_KEYBYTES] = {59, 208, 155, 110, 165, 227, 126, 84, 3, 115, 24, 123, 122, 145, 247, 72, 86, 87, 53, 64, 231, 83, 113, 86, 68, 184, 66, 118, 254, 255, 23, 101};
uint8_t sk_tv[SABER_SECRETKEYBYTES] = {1, 224, 255, 3, 0, 0, 224, 255, 5, 64, 0, 0, 0, 255, 63, 0, 0, 128, 0, 224, 255, 255, 255, 0, 248, 255, 254, 191, 255, 255, 255, 1, 16, 0, 4, 0, 0, 8, 0, 2, 64, 0, 252, 127, 0, 0, 0, 254, 127, 0, 0, 0, 254, 31, 0, 240, 127, 0, 16, 0, 0, 192, 255, 7, 0, 255, 63, 0, 0, 0, 255, 223, 255, 1, 128, 255, 239, 255, 255, 255, 255, 255, 127, 0, 240, 255, 255, 191, 255, 15, 0, 0, 0, 0, 252, 255, 255, 15, 0, 2, 64, 0, 16, 0, 0, 32, 0, 4, 0, 1, 240, 255, 1, 0, 0, 0, 0, 1, 32, 0, 4, 128, 1, 16, 0, 2, 192, 255, 255, 255, 253, 63, 0, 252, 255, 0, 240, 255, 1, 128, 255, 15, 0, 255, 63, 0, 248, 127, 0, 224, 255, 253, 127, 0, 0, 0, 0, 0, 0, 0, 128, 0, 16, 0, 254, 255, 255, 7, 0, 1, 0, 0, 252, 255, 0, 16, 0, 250, 191, 255, 7, 0, 3, 192, 255, 3, 0, 1, 224, 255, 1, 64, 0, 24, 0, 0, 32, 0, 4, 128, 0, 240, 255, 251, 191, 255, 7, 0, 254, 31, 0, 4, 128, 255, 255, 255, 253, 127, 0, 0, 0, 255, 95, 0, 252, 127, 0, 64, 0, 2, 128, 255, 31, 0, 255, 95, 0, 4, 0, 1, 48, 0, 0, 192, 255, 7, 0, 0, 192, 255, 255, 127, 0, 240, 255, 1, 128, 0, 248, 255, 1, 32, 0, 248, 255, 255, 223, 255, 1, 128, 0, 0, 0, 2, 0, 0, 8, 128, 0, 224, 255, 255, 191, 255, 7, 0, 0, 224, 255, 15, 0, 255, 15, 0, 0, 128, 255, 15, 0, 1, 32, 0, 4, 0, 0, 0, 0, 0, 128, 0, 248, 255, 255, 63, 0, 0, 128, 0, 240, 255, 3, 0, 0, 8, 0, 253, 223, 255, 3, 0, 0, 224, 255, 255, 255, 255, 7, 0, 1, 32, 0, 252, 255, 0, 16, 0, 4, 128, 255, 247, 255, 1, 192, 255, 11, 0, 0, 224, 255, 3, 192, 255, 7, 0, 255, 255, 255, 7, 128, 255, 239, 255, 5, 128, 255, 15, 0, 1, 0, 0, 0, 128, 255, 15, 0, 254, 255, 255, 15, 0, 2, 32, 0, 252, 127, 0, 16, 0, 4, 192, 255, 15, 0, 2, 32, 0, 4, 128, 255, 15, 0, 0, 192, 255, 15, 0, 255, 31, 0, 0, 128, 0, 240, 255, 1, 0, 0, 8, 0, 3, 224, 255, 7, 0, 0, 16, 0, 0, 128, 255, 15, 0, 1, 32, 0, 248, 255, 255, 47, 0, 254, 127, 255, 15, 0, 255, 63, 0, 0, 0, 0, 16, 0, 6, 128, 255, 239, 255, 0, 0, 0, 0, 128, 255, 15, 0, 252, 63, 0, 8, 0, 0, 0, 0, 0, 0, 0, 16, 0, 4, 0, 0, 8, 0, 253, 31, 0, 252, 127, 1, 224, 255, 253, 191, 0, 8, 0, 1, 64, 0, 8, 0, 0, 224, 255, 1, 0, 0, 240, 255, 1, 32, 0, 4, 0, 1, 16, 0, 2, 64, 0, 240, 255, 0, 32, 0, 8, 128, 255, 47, 0, 4, 192, 255, 255, 255, 255, 127, 0, 0, 0, 0, 224, 255, 3, 0, 0, 0, 0, 0, 32, 0, 0, 128, 0, 0, 0, 254, 255, 255, 7, 0, 1, 32, 0, 252, 255, 255, 15, 0, 252, 191, 0, 248, 255, 254, 31, 0, 252, 127, 1, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 252, 127, 0, 0, 0, 2, 0, 0, 248, 255, 0, 0, 0, 8, 128, 0, 16, 0, 254, 127, 0, 8, 0, 0, 96, 0, 252, 255, 0, 48, 0, 254, 255, 255, 23, 0, 0, 0, 0, 4, 0, 0, 240, 255, 7, 0, 0, 240, 255, 254, 31, 0, 248, 127, 1, 16, 0, 250, 191, 0, 8, 0, 1, 224, 255, 3, 0, 0, 0, 0, 254, 127, 0, 8, 0, 254, 95, 0, 8, 128, 255, 63, 0, 2, 0, 0, 0, 0, 2, 32, 0, 4, 0, 255, 255, 255, 3, 64, 0, 0, 0, 0, 224, 255, 7, 128, 255, 223, 255, 5, 192, 255, 247, 255, 255, 223, 255, 7, 128, 254, 31, 0, 2, 128, 0, 0, 0, 1, 32, 0, 0, 0, 0, 16, 0, 254, 255, 255, 255, 255, 1, 32, 0, 0, 128, 1, 16, 0, 0, 0, 0, 0, 0, 255, 31, 0, 8, 128, 255, 239, 255, 3, 192, 255, 23, 0, 0, 64, 0, 8, 128, 0, 16, 0, 250, 63, 0, 8, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 64, 0, 16, 0, 0, 0, 0, 4, 128, 0, 0, 0, 254, 255, 255, 15, 0, 0, 64, 0, 248, 127, 2, 240, 255, 253, 255, 255, 247, 255, 0, 224, 255, 11, 128, 0, 16, 0, 254, 63, 0, 240, 255, 254, 31, 0, 8, 128, 255, 15, 0, 252, 191, 255, 247, 255, 0, 64, 0, 0, 128, 255, 255, 255, 255, 127, 0, 0, 0, 2, 96, 0, 0, 0, 255, 31, 0, 250, 191, 0, 16, 0, 1, 32, 0, 252, 255, 255, 255, 255, 1, 0, 0, 0, 0, 0, 0, 0, 4, 128, 0, 16, 0, 252, 191, 255, 255, 255, 0, 64, 0, 244, 255, 0, 0, 0, 2, 64, 0, 8, 0, 255, 255, 255, 7, 128, 254, 47, 0, 254, 255, 255, 247, 255, 255, 95, 0, 0, 0, 255, 31, 0, 2, 192, 255, 15, 0, 252, 255, 255, 7, 0, 1, 16, 0, 0, 0, 0, 248, 255, 1, 0, 0, 4, 0, 0, 224, 255, 3, 64, 255, 15, 0, 1, 0, 0, 252, 127, 255, 31, 0, 254, 191, 0, 248, 255, 254, 31, 0, 8, 0, 255, 15, 0, 254, 63, 0, 8, 0, 1, 128, 255, 11, 128, 255, 239, 255, 3, 64, 0, 8, 0, 1, 0, 0, 252, 255, 255, 15, 0, 254, 191, 255, 7, 0, 254, 63, 0, 12, 0, 0, 224, 255, 5, 0, 0, 248, 255, 0, 224, 255, 3, 128, 255, 239, 255, 5, 128, 255, 255, 255, 255, 95, 0, 248, 127, 0, 208, 255, 255, 63, 0, 232, 255, 1, 32, 0, 252, 255, 255, 15, 0, 0, 192, 255, 247, 255, 254, 63, 0, 244, 255, 0, 224, 255, 253, 255, 255, 255, 255, 1, 224, 255, 3, 128, 0, 0, 0, 0, 0, 0, 248, 255, 0, 32, 0, 8, 128, 0, 240, 255, 7, 64, 0, 8, 0, 1, 32, 0, 248, 127, 0, 224, 255, 255, 63, 0, 240, 255, 254, 255, 255, 255, 255, 255, 31, 0, 0, 64, 0, 248, 255, 1, 32, 0, 4, 0, 0, 240, 255, 1, 64, 0, 8, 0, 255, 63, 0, 252, 127, 1, 16, 0, 4, 0, 0, 8, 0, 254, 63, 0, 4, 0, 0, 16, 0, 2, 0, 0, 240, 255, 2, 32, 0, 0, 0, 1, 48, 0, 254, 127, 0, 0, 0, 0, 32, 0, 4, 128, 0, 16, 0, 0, 128, 0, 0, 0, 255, 63, 0, 248, 127, 1, 16, 0, 254, 63, 0, 8, 0, 0, 192, 255, 255, 255, 255, 239, 255, 253, 255, 255, 247, 255, 2, 0, 0, 252, 127, 255, 15, 0, 252, 255, 255, 15, 0, 255, 255, 255, 7, 0, 0, 240, 255, 253, 127, 0, 16, 0, 235, 174, 169, 23, 122, 182, 12, 219, 33, 229, 182, 223, 150, 159, 25, 175, 115, 7, 31, 163, 5, 11, 76, 168, 144, 126, 155, 254, 56, 90, 231, 228, 233, 241, 182, 74, 191, 59, 6, 131, 97, 157, 18, 70, 211, 167, 36, 206, 44, 147, 196, 181, 119, 98, 216, 89, 71, 190, 162, 137, 105, 66, 232, 170, 30, 123, 178, 243, 255, 186, 81, 162, 150, 217, 127, 145, 51, 195, 168, 199, 77, 253, 24, 227, 141, 229, 181, 36, 14, 65, 232, 133, 145, 104, 5, 132, 36, 198, 92, 223, 156, 209, 144, 82, 2, 248, 138, 90, 247, 114, 31, 133, 6, 65, 116, 242, 249, 39, 185, 125, 105, 129, 206, 95, 38, 31, 162, 31, 250, 190, 116, 187, 59, 121, 179, 35, 165, 72, 34, 167, 5, 166, 166, 129, 87, 194, 238, 209, 243, 122, 159, 92, 105, 161, 213, 166, 103, 193, 186, 53, 212, 191, 111, 241, 210, 31, 55, 187, 9, 14, 147, 85, 221, 238, 75, 58, 135, 168, 204, 232, 56, 161, 176, 167, 246, 167, 71, 81, 75, 72, 178, 8, 152, 211, 100, 175, 204, 161, 50, 220, 15, 82, 251, 18, 7, 60, 35, 93, 40, 31, 123, 3, 52, 27, 10, 27, 11, 166, 238, 100, 220, 214, 214, 65, 56, 61, 130, 71, 30, 136, 42, 106, 149, 159, 142, 168, 66, 117, 87, 165, 163, 240, 238, 101, 5, 240, 212, 213, 44, 153, 5, 31, 182, 254, 194, 169, 78, 115, 97, 131, 103, 38, 184, 169, 39, 244, 88, 168, 157, 24, 208, 88, 159, 158, 194, 251, 40, 25, 225, 87, 111, 151, 114, 236, 11, 98, 140, 41, 232, 125, 252, 53, 237, 162, 106, 186, 29, 213, 102, 169, 202, 109, 186, 5, 81, 156, 205, 207, 100, 9, 34, 91, 24, 118, 21, 63, 67, 202, 11, 45, 239, 150, 151, 165, 165, 58, 68, 155, 70, 225, 234, 167, 188, 63, 5, 78, 93, 168, 58, 182, 140, 8, 110, 160, 201, 169, 140, 50, 41, 101, 233, 180, 70, 109, 82, 198, 27, 131, 198, 35, 69, 238, 162, 139, 225, 76, 181, 201, 111, 69, 71, 30, 247, 239, 96, 118, 175, 227, 223, 63, 167, 168, 136, 153, 198, 46, 50, 191, 220, 191, 223, 136, 151, 100, 69, 19, 128, 140, 16, 162, 9, 191, 250, 56, 247, 234, 148, 230, 172, 15, 19, 155, 195, 184, 173, 255, 196, 175, 54, 116, 86, 48, 190, 61, 167, 80, 56, 119, 126, 151, 156, 254, 192, 196, 189, 25, 71, 14, 28, 220, 47, 41, 134, 220, 178, 66, 251, 112, 127, 86, 51, 46, 117, 59, 13, 185, 142, 104, 193, 8, 151, 135, 194, 241, 219, 179, 63, 142, 2, 28, 82, 137, 214, 68, 13, 101, 97, 39, 79, 230, 255, 20, 134, 1, 191, 201, 57, 58, 26, 80, 169, 223, 26, 17, 162, 116, 71, 128, 107, 240, 73, 18, 233, 194, 16, 60, 53, 231, 186, 142, 201, 67, 50, 180, 218, 174, 160, 215, 159, 154, 33, 173, 126, 69, 9, 87, 177, 70, 141, 59, 164, 71, 119, 230, 94, 79, 52, 180, 16, 128, 206, 144, 13, 59, 203, 70, 235, 105, 113, 162, 81, 70, 189, 91, 169, 29, 12, 172, 225, 200, 48, 59, 98, 16, 241, 188, 230, 73, 204, 253, 146, 237, 16, 97, 206, 138, 70, 143, 230, 137, 200, 229, 222, 226, 140, 45, 92, 247, 26, 136, 193, 179, 253, 66, 130, 3, 155, 151, 41, 13, 233, 35, 220, 90, 2, 224, 206, 28, 111, 110, 175, 84, 80, 75, 34, 122, 68, 24, 60, 25, 74, 121, 156, 245, 112, 251, 1, 125, 106, 82, 187, 136, 223, 248, 98, 192, 253, 179, 23, 223, 219, 91, 58, 76, 87, 154, 153, 0, 35, 71, 5, 160, 205, 135, 138, 106, 35, 241, 233, 6, 75, 203, 21, 142, 146, 215, 228, 104, 117, 95, 60, 233, 109, 44, 217, 36, 14, 147, 233, 107, 239, 198, 115, 66, 35, 111, 75, 183, 254, 84, 53, 165, 146, 240, 149, 109, 242, 112, 217, 137, 194, 28, 9, 99, 94, 198, 90, 201, 178, 251, 66, 173, 171, 214, 164, 137, 227, 168, 173, 251, 241, 59, 197, 94, 150, 126, 114, 186, 165, 205, 204, 133, 4, 245, 173, 94, 225, 128, 204, 255, 86, 143, 12, 188, 207, 51, 2, 254, 210, 144, 219, 69, 17, 173, 129, 18, 105, 207, 65, 236, 156, 174, 14, 91, 146, 177, 84, 241, 198, 213, 210, 56, 77, 246, 41, 207, 170, 0, 218, 192, 84, 126, 47, 211, 124, 254, 227, 218, 215, 213, 160, 51, 246, 55, 204, 205, 174, 83, 77, 224, 9, 230, 146, 42, 61, 135, 214, 23, 118, 102, 70, 82, 136, 92, 165, 94, 179, 154, 171, 187, 171, 236, 102, 169, 205, 113, 130, 45, 72, 95, 61, 192, 160, 246, 221, 104, 89, 37, 191, 38, 52, 228, 136, 83, 194, 44, 192, 124, 177, 26, 63, 94, 185, 61, 7, 219, 185, 132, 151, 99, 52, 1, 232, 234, 22, 7, 176, 63, 12, 119, 216, 3, 239, 126, 97, 253, 34, 31, 192, 178, 58, 133, 54, 36, 217, 47, 149, 214, 6, 94, 120, 178, 236, 71, 172, 144, 35, 193, 220, 192, 107, 39, 201, 158, 96, 73, 181, 254, 103, 246, 75, 145, 48, 231, 65, 118, 92, 24, 173, 169, 190, 247, 41, 111, 74, 235, 92, 116, 94, 25, 76, 169, 45, 187, 214, 2, 240, 244, 54, 198, 12, 241, 232, 230, 45, 1, 249, 94, 205, 159, 203, 160, 79, 111, 60, 181, 193, 191, 126, 57, 7, 136, 0, 167, 167, 102, 190, 94, 36, 74, 211, 252, 226, 158, 57, 131, 156, 254, 130, 201, 170, 171, 242, 124, 207, 47, 198, 83, 86, 45, 147, 143, 186, 121, 20, 115, 190, 206, 151, 233, 172, 241, 139, 230, 135, 57, 123, 24, 96, 118, 75, 170, 78, 151, 243, 85, 223, 167, 240, 200, 202, 0, 58, 195, 178, 240, 4, 245, 67, 95, 16, 196, 205, 69, 17, 72, 68, 122, 253, 155, 153, 178, 9, 119, 13, 224, 208, 58, 205, 183, 188, 107, 229, 113, 104, 140};

uint8_t buf_tv[64] = {120, 151, 113, 128, 66, 173, 1, 11, 201, 139, 233, 93, 19, 221, 222, 240, 101, 51, 171, 149, 66, 111, 175, 199, 73, 118, 205, 153, 173, 183, 69, 98, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};