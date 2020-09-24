#note the treedata below is not in the order output by semantic seg nets. I eventually
#found the csv of top level classes in (presumably) the correct order on the scene
#challenge website. in any event the ade20k nets are too noisy.

treeData = [ #lifted from ade20k website source
{ "name": "root",
"children": [
{ "name": "air conditioner, air conditioning (582)"},
{ "name": "airplane, aeroplane, plane (317)",
"children": [
{ "name": "fuselage (59)",
"children": [
{ "name": "cockpit (32)"},
{ "name": "door (34)"},
{ "name": "window (46)"},
{ "name": "windows (15)"}
]
},
{ "name": "landing gear (114)",
"children": [
{ "name": "wheel (13)"}
]
},
{ "name": "stabilizer (91)"},
{ "name": "turbine engine (53)"},
{ "name": "wing (68)"}
]
},
{ "name": "animal, animate being, beast, brute, creature, fauna (533)"},
{ "name": "apparel, wearing apparel, dress, clothes (466)"},
{ "name": "armchair (2320)",
"children": [
{ "name": "apron (158)"},
{ "name": "arm (1515)",
"children": [
{ "name": "arm panel (123)"},
{ "name": "arm support (153)"},
{ "name": "inside arm (533)"},
{ "name": "manchette (47)"},
{ "name": "outside arm (340)"}
]
},
{ "name": "back (575)",
"children": [
{ "name": "rail (16)"},
{ "name": "spindle (14)"},
{ "name": "stile (18)"}
]
},
{ "name": "back pillow (353)"},
{ "name": "earmuffs (102)"},
{ "name": "leg (948)"},
{ "name": "seat (227)"},
{ "name": "seat base (310)"},
{ "name": "seat cushion (547)"},
{ "name": "skirt (83)"},
{ "name": "stretcher (13)"}
]
},
{ "name": "ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin (950)"},
{ "name": "awning, sunshade, sunblind (1024)"},
{ "name": "bag (1170)"},
{ "name": "bag, handbag, pocketbook, purse (329)"},
{ "name": "bag, traveling bag, travelling bag, grip, suitcase (276)"},
{ "name": "ball (733)"},
{ "name": "bannister, banister, balustrade, balusters, handrail (785)"},
{ "name": "barrel, cask (322)"},
{ "name": "base, pedestal, stand (767)"},
{ "name": "basket, handbasket (1141)"},
{ "name": "bathtub, bathing tub, bath, tub (443)",
"children": [
{ "name": "faucet (186)"},
{ "name": "overflot plate (14)"},
{ "name": "tap (41)"}
]
},
{ "name": "beam (351)"},
{ "name": "bed (2418)",
"children": [
{ "name": "base (21)"},
{ "name": "bedpost (55)"},
{ "name": "bedspring (10)"},
{ "name": "drawer (10)"},
{ "name": "footboard (535)"},
{ "name": "headboard (1186)"},
{ "name": "ladder (22)"},
{ "name": "leg (564)"},
{ "name": "rail (19)"},
{ "name": "safety rail (17)"},
{ "name": "side (26)"},
{ "name": "side rail (107)"}
]
},
{ "name": "bench (1352)",
"children": [
{ "name": "leg (27)"}
]
},
{ "name": "bicycle, bike, wheel, cycle (621)"},
{ "name": "blanket, cover (278)"},
{ "name": "blind, screen (700)",
"children": [
{ "name": "head rail (21)"},
{ "name": "slats (20)"}
]
},
{ "name": "board, plank (327)"},
{ "name": "boat (804)",
"children": [
{ "name": "window (11)"}
]
},
{ "name": "book (5083)"},
{ "name": "bookcase (529)",
"children": [
{ "name": "door (41)",
"children": [
{ "name": "knob (14)"}
]
},
{ "name": "front (11)"},
{ "name": "shelf (201)"},
{ "name": "top (12)"}
]
},
{ "name": "booklet, brochure, folder, leaflet, pamphlet (262)"},
{ "name": "bottle (3643)",
"children": [
{ "name": "base (16)"},
{ "name": "cap (34)"},
{ "name": "label (22)"},
{ "name": "neck (17)"}
]
},
{ "name": "bowl (938)",
"children": [
{ "name": "bowl (12)"},
{ "name": "opening (14)"}
]
},
{ "name": "box (4358)",
"children": [
{ "name": "tissue (18)"}
]
},
{ "name": "bridge, span (318)"},
{ "name": "bucket, pail (503)"},
{ "name": "building, edifice (18850)",
"children": [
{ "name": "arcades (42)"},
{ "name": "balcony (2060)",
"children": [
{ "name": "railing (31)"},
{ "name": "shutter (51)"}
]
},
{ "name": "balustrade (85)"},
{ "name": "bars (10)"},
{ "name": "bell (34)"},
{ "name": "chimney (425)"},
{ "name": "column (1053)",
"children": [
{ "name": "base (60)"},
{ "name": "capital (87)"},
{ "name": "shaft (91)"}
]
},
{ "name": "dome (122)"},
{ "name": "door (2934)",
"children": [
{ "name": "handle (18)"},
{ "name": "pane (58)"}
]
},
{ "name": "door frame (19)"},
{ "name": "doors (14)"},
{ "name": "dormer (279)",
"children": [
{ "name": "window (24)"}
]
},
{ "name": "double door (324)",
"children": [
{ "name": "door (18)",
"children": [
{ "name": "pane (12)"}
]
}
]
},
{ "name": "entrance (106)"},
{ "name": "fire escape (75)"},
{ "name": "garage door (40)"},
{ "name": "gate (15)"},
{ "name": "grille (59)"},
{ "name": "metal shutter (228)"},
{ "name": "metal shutters (48)"},
{ "name": "pane (89)"},
{ "name": "pipe (122)"},
{ "name": "porch (10)"},
{ "name": "railing (489)"},
{ "name": "revolving door (18)"},
{ "name": "roof (739)",
"children": [
{ "name": "rakes (10)"}
]
},
{ "name": "shop window (755)"},
{ "name": "shutter (570)"},
{ "name": "sign (10)"},
{ "name": "skylight (14)"},
{ "name": "statue (24)"},
{ "name": "steps (47)"},
{ "name": "terrace (49)"},
{ "name": "tower (19)"},
{ "name": "wall (40)"},
{ "name": "window (35737)",
"children": [
{ "name": "casing (26)"},
{ "name": "lower sash (17)",
"children": [
{ "name": "pane (22)"},
{ "name": "rail (13)"},
{ "name": "stile (13)"}
]
},
{ "name": "muntin (77)"},
{ "name": "pane (354)"},
{ "name": "rail (46)"},
{ "name": "sash (13)",
"children": [
{ "name": "pane (16)"},
{ "name": "rail (26)"},
{ "name": "stile (20)"}
]
},
{ "name": "shutter (275)"},
{ "name": "stile (38)"},
{ "name": "upper sash (14)",
"children": [
{ "name": "pane (16)"},
{ "name": "rail (14)"},
{ "name": "stile (13)"}
]
},
{ "name": "window (83)"}
]
},
{ "name": "windows (303)"}
]
},
{ "name": "bulletin board, notice board (308)"},
{ "name": "bus, autobus, coach, charabanc, double-decker, jitney, motorbus, motorcoach, omnibus, passenger vehicle (477)",
"children": [
{ "name": "door (25)"},
{ "name": "headlight (33)"},
{ "name": "license plate (21)"},
{ "name": "mirror (15)"},
{ "name": "taillight (12)"},
{ "name": "wheel (63)",
"children": [
{ "name": "rim (14)"}
]
},
{ "name": "window (169)"},
{ "name": "windshield (40)"}
]
},
{ "name": "cabinet (7417)",
"children": [
{ "name": "back (11)"},
{ "name": "base (20)"},
{ "name": "door (4685)",
"children": [
{ "name": "handle (1483)"},
{ "name": "hinge (239)"},
{ "name": "knob (1916)"},
{ "name": "muntin (211)"},
{ "name": "pane (611)"},
{ "name": "window (15)"}
]
},
{ "name": "drawer (2092)",
"children": [
{ "name": "handle (932)"},
{ "name": "knob (924)"}
]
},
{ "name": "front (334)"},
{ "name": "leg (59)"},
{ "name": "panel (14)"},
{ "name": "shelf (144)"},
{ "name": "side (289)"},
{ "name": "skirt (142)"},
{ "name": "top (364)"}
]
},
{ "name": "can, tin, tin can (556)"},
{ "name": "candle, taper, wax light (326)"},
{ "name": "candlestick, candle holder (775)",
"children": [
{ "name": "candle (11)"}
]
},
{ "name": "car, auto, automobile, machine, motorcar (14977)",
"children": [
{ "name": "bumper (661)"},
{ "name": "door (1696)",
"children": [
{ "name": "handle (1398)"},
{ "name": "mirror (640)"},
{ "name": "window (1624)"}
]
},
{ "name": "fender (19)"},
{ "name": "gas cap (74)"},
{ "name": "handle (10)"},
{ "name": "headlight (1363)"},
{ "name": "hood (15)"},
{ "name": "license plate (1324)"},
{ "name": "logo (19)"},
{ "name": "mirror (365)"},
{ "name": "roof rack (10)"},
{ "name": "taillight (1315)"},
{ "name": "wheel (3456)",
"children": [
{ "name": "rim (1925)"}
]
},
{ "name": "window (1948)"},
{ "name": "windshield (909)"},
{ "name": "wiper (26)"}
]
},
{ "name": "ceiling (8516)",
"children": [
{ "name": "beam (199)"}
]
},
{ "name": "central reservation (281)"},
{ "name": "chair (13766)",
"children": [
{ "name": "apron (546)"},
{ "name": "arm (415)",
"children": [
{ "name": "arm support (93)"},
{ "name": "armrest (16)"},
{ "name": "manchette (18)"}
]
},
{ "name": "back (1468)",
"children": [
{ "name": "rail (529)"},
{ "name": "spindle (454)"},
{ "name": "stile (556)"}
]
},
{ "name": "back pillow (64)"},
{ "name": "base (10)"},
{ "name": "foot rest (11)"},
{ "name": "h-stretcher (36)"},
{ "name": "leg (2911)"},
{ "name": "seat (1059)"},
{ "name": "seat base (31)"},
{ "name": "seat cushion (152)"},
{ "name": "skirt (51)"},
{ "name": "stretcher (277)"}
]
},
{ "name": "chandelier, pendant, pendent (811)",
"children": [
{ "name": "arm (152)"},
{ "name": "bulb (157)"},
{ "name": "canopy (58)"},
{ "name": "chain (65)"},
{ "name": "shade (611)"}
]
},
{ "name": "chest of drawers, chest, bureau, dresser (663)",
"children": [
{ "name": "base (19)"},
{ "name": "door (18)"},
{ "name": "drawer (1777)",
"children": [
{ "name": "handle (930)"},
{ "name": "knob (857)"}
]
},
{ "name": "front (36)"},
{ "name": "leg (35)"},
{ "name": "mirror (10)"},
{ "name": "side (44)"},
{ "name": "skirt (40)"},
{ "name": "top (35)"}
]
},
{ "name": "clock (947)",
"children": [
{ "name": "face (179)",
"children": [
{ "name": "hand (316)"}
]
}
]
},
{ "name": "coffee table, cocktail table (935)",
"children": [
{ "name": "apron (96)"},
{ "name": "drawer (25)",
"children": [
{ "name": "knob (13)"}
]
},
{ "name": "leg (419)"},
{ "name": "shelf (27)"},
{ "name": "top (167)"}
]
},
{ "name": "column, pillar (2009)",
"children": [
{ "name": "base (64)"},
{ "name": "capital (91)"},
{ "name": "shaft (92)"}
]
},
{ "name": "computer, computing machine, computing device, data processor, electronic computer, information processing system (555)",
"children": [
{ "name": "computer case (232)"},
{ "name": "keyboard (389)",
"children": [
{ "name": "keys (33)"}
]
},
{ "name": "monitor (454)",
"children": [
{ "name": "screen (410)"}
]
},
{ "name": "mouse (229)"},
{ "name": "speaker (88)"}
]
},
{ "name": "container (354)"},
{ "name": "counter (467)"},
{ "name": "countertop (381)"},
{ "name": "cup (308)"},
{ "name": "curb, curbing, kerb (331)"},
{ "name": "curtain, drape, drapery, mantle, pall (4469)"},
{ "name": "cushion (4906)"},
{ "name": "deck chair, beach chair (251)"},
{ "name": "desk (1326)",
"children": [
{ "name": "door (21)"},
{ "name": "drawer (264)",
"children": [
{ "name": "handle (85)"},
{ "name": "knob (92)"}
]
},
{ "name": "leg (47)"},
{ "name": "shelf (11)"},
{ "name": "side (17)"},
{ "name": "top (33)"}
]
},
{ "name": "dishwasher, dish washer, dishwashing machine (270)",
"children": [
{ "name": "button panel (35)",
"children": [
{ "name": "buttons (10)"},
{ "name": "dial (15)"}
]
},
{ "name": "door (35)"}
]
},
{ "name": "door (4672)",
"children": [
{ "name": "door frame (591)"},
{ "name": "handle (317)"},
{ "name": "hinge (44)"},
{ "name": "knob (284)"},
{ "name": "lock (46)"},
{ "name": "muntin (56)"},
{ "name": "pane (269)"},
{ "name": "panel (44)"},
{ "name": "window (37)"}
]
},
{ "name": "doorframe, doorcase (336)"},
{ "name": "double door (471)",
"children": [
{ "name": "door (291)",
"children": [
{ "name": "handle (58)"},
{ "name": "hinge (30)"},
{ "name": "knob (42)"},
{ "name": "lock (21)"},
{ "name": "muntin (129)"},
{ "name": "pane (381)"},
{ "name": "panel (11)"}
]
},
{ "name": "door frame (103)"},
{ "name": "handle (32)"},
{ "name": "pane (22)"}
]
},
{ "name": "earth, ground (2592)"},
{ "name": "fan (492)",
"children": [
{ "name": "blade (460)"},
{ "name": "canopy (39)"},
{ "name": "motor (53)"},
{ "name": "shade (22)"},
{ "name": "tube (17)"}
]
},
{ "name": "fence, fencing (2278)",
"children": [
{ "name": "post (27)"},
{ "name": "rail (32)"}
]
},
{ "name": "field (776)",
"children": [
{ "name": "hay bale (10)"}
]
},
{ "name": "figurine, statuette (623)"},
{ "name": "fireplace, hearth, open fireplace (508)"},
{ "name": "flag (909)"},
{ "name": "floor, flooring (11172)"},
{ "name": "flower (2404)"},
{ "name": "fluorescent, fluorescent fixture (1101)",
"children": [
{ "name": "backplate (13)"},
{ "name": "diffusor (15)"}
]
},
{ "name": "food, solid food (799)"},
{ "name": "fruit (794)"},
{ "name": "gate (336)"},
{ "name": "glass, drinking glass (1709)",
"children": [
{ "name": "base (103)"},
{ "name": "bowl (115)"},
{ "name": "opening (117)"},
{ "name": "stem (106)"}
]
},
{ "name": "grass (4253)"},
{ "name": "gravestone, headstone, tombstone (435)"},
{ "name": "grill, grille, grillwork (252)"},
{ "name": "hat, chapeau, lid (361)"},
{ "name": "hedge, hedgerow (298)"},
{ "name": "hill (374)"},
{ "name": "hood, exhaust hood (295)",
"children": [
{ "name": "body (37)"},
{ "name": "filter (19)"},
{ "name": "vent (39)"}
]
},
{ "name": "house (1227)",
"children": [
{ "name": "balcony (43)"},
{ "name": "balustrade (12)"},
{ "name": "chimney (134)"},
{ "name": "column (97)",
"children": [
{ "name": "base (12)"},
{ "name": "capital (20)"},
{ "name": "shaft (20)"}
]
},
{ "name": "door (353)",
"children": [
{ "name": "pane (20)"}
]
},
{ "name": "dormer (37)"},
{ "name": "double door (22)"},
{ "name": "garage door (43)"},
{ "name": "pipe (13)"},
{ "name": "railing (107)",
"children": [
{ "name": "post (16)"},
{ "name": "rail (15)"}
]
},
{ "name": "roof (280)",
"children": [
{ "name": "rakes (16)"}
]
},
{ "name": "shutter (73)"},
{ "name": "steps (12)"},
{ "name": "window (1665)",
"children": [
{ "name": "casing (15)"},
{ "name": "muntin (36)"},
{ "name": "pane (109)"},
{ "name": "shutter (18)"}
]
},
{ "name": "windows (10)"}
]
},
{ "name": "jar (1013)",
"children": [
{ "name": "label (11)"}
]
},
{ "name": "lamp (6036)",
"children": [
{ "name": "aperture (11)"},
{ "name": "arm (140)"},
{ "name": "base (1021)"},
{ "name": "bulb (85)"},
{ "name": "canopy (202)"},
{ "name": "chain (98)"},
{ "name": "column (1527)"},
{ "name": "cord (161)"},
{ "name": "shade (2582)"},
{ "name": "tube (231)"}
]
},
{ "name": "land, ground, soil (275)"},
{ "name": "light, light source (10353)",
"children": [
{ "name": "aperture (455)"},
{ "name": "backplate (65)"},
{ "name": "bulb (13)"},
{ "name": "canopy (53)"},
{ "name": "diffusor (376)"},
{ "name": "shade (192)"}
]
},
{ "name": "loudspeaker, speaker, speaker unit, loudspeaker system, speaker system (330)"},
{ "name": "magazine (597)"},
{ "name": "manhole (386)"},
{ "name": "microwave, microwave oven (374)",
"children": [
{ "name": "button (14)"},
{ "name": "button panel (56)",
"children": [
{ "name": "screen (13)"}
]
},
{ "name": "buttons (18)"},
{ "name": "dial (24)"},
{ "name": "display (15)"},
{ "name": "door (109)",
"children": [
{ "name": "handle (44)"},
{ "name": "window (32)"}
]
},
{ "name": "screen (23)"}
]
},
{ "name": "minibike, motorbike (674)",
"children": [
{ "name": "license plate (10)"},
{ "name": "wheel (10)"}
]
},
{ "name": "mirror (1598)",
"children": [
{ "name": "frame (183)"}
]
},
{ "name": "monitor, monitoring device (601)",
"children": [
{ "name": "screen (192)"}
]
},
{ "name": "mountain, mount (3012)"},
{ "name": "mug (395)",
"children": [
{ "name": "bowl (24)"},
{ "name": "handle (157)"},
{ "name": "opening (26)"}
]
},
{ "name": "napkin, table napkin, serviette (464)"},
{ "name": "ottoman, pouf, pouffe, puff, hassock (433)",
"children": [
{ "name": "leg (99)"},
{ "name": "seat (16)"},
{ "name": "seat base (37)"},
{ "name": "seat cushion (64)"}
]
},
{ "name": "oven (272)",
"children": [
{ "name": "button panel (55)",
"children": [
{ "name": "buttons (12)"},
{ "name": "dial (85)"},
{ "name": "screen (21)"}
]
},
{ "name": "dial (26)"},
{ "name": "door (87)",
"children": [
{ "name": "handle (76)"},
{ "name": "window (21)"}
]
}
]
},
{ "name": "painting, picture (8869)",
"children": [
{ "name": "frame (694)"}
]
},
{ "name": "palm, palm tree (1200)"},
{ "name": "pane, pane of glass, window glass (397)"},
{ "name": "paper (782)"},
{ "name": "path (747)"},
{ "name": "patty, cake (393)"},
{ "name": "person, individual, someone, somebody, mortal, soul (24420)",
"children": [
{ "name": "back (581)"},
{ "name": "head (2833)",
"children": [
{ "name": "ear (189)"},
{ "name": "eye (1225)"},
{ "name": "hair (164)"},
{ "name": "mouth (720)",
"children": [
{ "name": "teeth (14)"}
]
},
{ "name": "nose (168)"}
]
},
{ "name": "left arm (1446)"},
{ "name": "left foot (356)"},
{ "name": "left hand (549)"},
{ "name": "left leg (1370)"},
{ "name": "left shoulder (13)"},
{ "name": "neck (175)"},
{ "name": "right arm (1498)"},
{ "name": "right foot (341)"},
{ "name": "right hand (663)"},
{ "name": "right leg (1335)"},
{ "name": "right shoulder (15)"},
{ "name": "torso (583)"}
]
},
{ "name": "pillow (2246)"},
{ "name": "pipe, pipage, piping (285)"},
{ "name": "pitcher, ewer (259)",
"children": [
{ "name": "handle (45)"}
]
},
{ "name": "place mat (252)"},
{ "name": "plant, flora, plant life (10967)",
"children": [
{ "name": "flower (18)"},
{ "name": "leaf (10)"}
]
},
{ "name": "plate (1760)"},
{ "name": "plaything, toy (1065)"},
{ "name": "pole (2163)"},
{ "name": "pool ball (319)"},
{ "name": "pool table, billiard table, snooker table (264)",
"children": [
{ "name": "base (36)",
"children": [
{ "name": "leg (18)"}
]
},
{ "name": "bed (86)"},
{ "name": "cabinet (45)",
"children": [
{ "name": "ball storage (11)"}
]
},
{ "name": "corner pocket (291)"},
{ "name": "leg (227)"},
{ "name": "rail (78)"},
{ "name": "side pocket (149)"}
]
},
{ "name": "poster, posting, placard, notice, bill, card (706)"},
{ "name": "pot (1212)"},
{ "name": "pot, flowerpot (2517)"},
{ "name": "railing, rail (1429)"},
{ "name": "refrigerator, icebox (460)",
"children": [
{ "name": "door (299)",
"children": [
{ "name": "handle (217)"},
{ "name": "ice maker (42)"}
]
},
{ "name": "side (12)"}
]
},
{ "name": "river (384)"},
{ "name": "road, route (5396)",
"children": [
{ "name": "crosswalk (363)"}
]
},
{ "name": "rock, stone (2873)"},
{ "name": "rug, carpet, carpeting (1512)"},
{ "name": "sand (423)"},
{ "name": "sconce (1912)",
"children": [
{ "name": "arm (209)"},
{ "name": "backplate (253)"},
{ "name": "bulb (59)"},
{ "name": "shade (485)"}
]
},
{ "name": "screen, crt screen (441)"},
{ "name": "sculpture (536)"},
{ "name": "sea (787)",
"children": [
{ "name": "wave (33)"}
]
},
{ "name": "seat (1808)",
"children": [
{ "name": "back (44)"},
{ "name": "back pillow (18)"},
{ "name": "seat cushion (81)"}
]
},
{ "name": "shelf (2403)",
"children": [
{ "name": "base (10)"},
{ "name": "door (61)",
"children": [
{ "name": "knob (12)"}
]
},
{ "name": "leg (11)"},
{ "name": "shelf (525)"},
{ "name": "side (43)"},
{ "name": "top (22)"}
]
},
{ "name": "shoe (810)"},
{ "name": "shrub, bush (1062)"},
{ "name": "sidewalk, pavement (5466)"},
{ "name": "signboard, sign (6108)"},
{ "name": "sink (1480)",
"children": [
{ "name": "bowl (58)"},
{ "name": "faucet (1106)"},
{ "name": "pedestal (29)"},
{ "name": "tap (147)"}
]
},
{ "name": "sky (9487)",
"children": [
{ "name": "cloud (273)"},
{ "name": "clouds (64)"}
]
},
{ "name": "skyscraper (970)",
"children": [
{ "name": "pane (12)"},
{ "name": "window (155)"}
]
},
{ "name": "snow (327)"},
{ "name": "sofa, couch, lounge (1710)",
"children": [
{ "name": "apron (56)"},
{ "name": "arm (1294)",
"children": [
{ "name": "arm panel (205)"},
{ "name": "arm support (10)"},
{ "name": "inside arm (579)"},
{ "name": "outside arm (343)"}
]
},
{ "name": "back (412)"},
{ "name": "back pillow (1458)"},
{ "name": "cushion (13)"},
{ "name": "leg (500)"},
{ "name": "seat (90)"},
{ "name": "seat base (482)"},
{ "name": "seat cushion (1553)"},
{ "name": "skirt (116)"}
]
},
{ "name": "spotlight, spot (2647)",
"children": [
{ "name": "arm (11)"},
{ "name": "backplate (11)"},
{ "name": "shade (170)"}
]
},
{ "name": "stairs, steps (1197)",
"children": [
{ "name": "step (38)",
"children": [
{ "name": "riser (37)"},
{ "name": "tread (30)"}
]
}
]
},
{ "name": "stairway, staircase (759)",
"children": [
{ "name": "rung (10)"},
{ "name": "step (104)",
"children": [
{ "name": "riser (84)"},
{ "name": "tread (82)"}
]
},
{ "name": "stringer (10)"}
]
},
{ "name": "step, stair (350)"},
{ "name": "stool (994)",
"children": [
{ "name": "apron (13)"},
{ "name": "footrest (11)"},
{ "name": "leg (223)"},
{ "name": "seat (80)"},
{ "name": "stretcher (68)"}
]
},
{ "name": "stove, kitchen stove, range, kitchen range, cooking stove (597)",
"children": [
{ "name": "burner (83)"},
{ "name": "button panel (207)",
"children": [
{ "name": "buttons (10)"},
{ "name": "dial (458)"},
{ "name": "screen (46)"}
]
},
{ "name": "dial (97)"},
{ "name": "drawer (74)",
"children": [
{ "name": "handle (25)"}
]
},
{ "name": "oven (273)",
"children": [
{ "name": "door (134)",
"children": [
{ "name": "handle (126)"},
{ "name": "window (74)"}
]
}
]
},
{ "name": "stove (268)",
"children": [
{ "name": "burner (212)"},
{ "name": "dial (11)"}
]
}
]
},
{ "name": "streetlight, street lamp (4660)",
"children": [
{ "name": "lamp housing (17)"}
]
},
{ "name": "switch, electric switch, electrical switch (711)",
"children": [
{ "name": "switch (17)"}
]
},
{ "name": "swivel chair (1084)",
"children": [
{ "name": "arm (99)",
"children": [
{ "name": "arm support (18)"},
{ "name": "armrest (32)"}
]
},
{ "name": "armrest (19)"},
{ "name": "back (117)"},
{ "name": "base (110)",
"children": [
{ "name": "wheel (329)"}
]
},
{ "name": "piston (60)"},
{ "name": "seat (103)"}
]
},
{ "name": "table (7988)",
"children": [
{ "name": "apron (159)"},
{ "name": "base (56)"},
{ "name": "door (71)",
"children": [
{ "name": "handle (20)"},
{ "name": "knob (29)"}
]
},
{ "name": "drawer (998)",
"children": [
{ "name": "handle (440)"},
{ "name": "knob (446)"}
]
},
{ "name": "front (40)"},
{ "name": "leg (1298)"},
{ "name": "pedestal (38)"},
{ "name": "shelf (22)"},
{ "name": "side (42)"},
{ "name": "skirt (22)"},
{ "name": "stretcher (14)"},
{ "name": "top (824)"}
]
},
{ "name": "telephone, phone, telephone set (525)",
"children": [
{ "name": "base (33)"},
{ "name": "buttons (18)"},
{ "name": "cord (17)"},
{ "name": "keyboard (15)"},
{ "name": "receiver (75)"},
{ "name": "screen (18)"}
]
},
{ "name": "television receiver, television, television set, tv, tv set, idiot box, boob tube, telly, goggle box (693)",
"children": [
{ "name": "screen (35)"}
]
},
{ "name": "text, textual matter (438)"},
{ "name": "toilet, can, commode, crapper, pot, potty, stool, throne (462)",
"children": [
{ "name": "bowl (105)"},
{ "name": "cistern (246)",
"children": [
{ "name": "flusher (58)"},
{ "name": "tank lid (39)"}
]
},
{ "name": "lid (325)"}
]
},
{ "name": "towel (1134)"},
{ "name": "trade name, brand name, brand, marque (705)"},
{ "name": "traffic light, traffic signal, stoplight (1120)",
"children": [
{ "name": "housing (163)",
"children": [
{ "name": "visor (15)"}
]
},
{ "name": "pole (11)"}
]
},
{ "name": "tray (767)"},
{ "name": "tree (22236)",
"children": [
{ "name": "branch (39)"},
{ "name": "fruit (14)"},
{ "name": "trunk (98)"}
]
},
{ "name": "truck, motortruck (672)",
"children": [
{ "name": "headlight (26)"},
{ "name": "license plate (15)"},
{ "name": "mirror (13)"},
{ "name": "wheel (65)"},
{ "name": "window (17)"},
{ "name": "windshield (30)"}
]
},
{ "name": "umbrella (543)"},
{ "name": "van (814)",
"children": [
{ "name": "door (77)",
"children": [
{ "name": "handle (44)"},
{ "name": "mirror (13)"},
{ "name": "window (62)"}
]
},
{ "name": "headlight (29)"},
{ "name": "license plate (44)"},
{ "name": "mirror (15)"},
{ "name": "taillight (43)"},
{ "name": "wheel (118)",
"children": [
{ "name": "rim (43)"}
]
},
{ "name": "window (92)"},
{ "name": "windshield (36)"}
]
},
{ "name": "vase (2121)"},
{ "name": "wall (33717)",
"children": [
{ "name": "pane (13)"}
]
},
{ "name": "wall socket, wall plug, electric outlet, electrical outlet, outlet, electric receptacle (1633)",
"children": [
{ "name": "plate (22)"},
{ "name": "sockets (25)"}
]
},
{ "name": "wardrobe, closet, press (429)",
"children": [
{ "name": "door (368)",
"children": [
{ "name": "handle (116)"},
{ "name": "hinge (24)"},
{ "name": "knob (73)"},
{ "name": "mirror (27)"}
]
},
{ "name": "drawer (150)",
"children": [
{ "name": "handle (76)"},
{ "name": "knob (44)"}
]
},
{ "name": "leg (12)"},
{ "name": "shelf (33)"},
{ "name": "side (17)"},
{ "name": "top (16)"}
]
},
{ "name": "water (885)"},
{ "name": "windowpane, window (9723)",
"children": [
{ "name": "casing (86)"},
{ "name": "door (10)"},
{ "name": "handle (14)"},
{ "name": "interior casing (11)"},
{ "name": "lower sash (179)",
"children": [
{ "name": "muntin (125)"},
{ "name": "pane (328)"},
{ "name": "rail (174)"},
{ "name": "stile (134)"}
]
},
{ "name": "muntin (134)"},
{ "name": "pane (1014)"},
{ "name": "rail (97)"},
{ "name": "sash (292)",
"children": [
{ "name": "handle (13)"},
{ "name": "muntin (196)"},
{ "name": "pane (546)"},
{ "name": "rail (186)"},
{ "name": "sash lock (31)"},
{ "name": "stile (175)"}
]
},
{ "name": "sash lock (10)"},
{ "name": "shutter (14)"},
{ "name": "sill (69)"},
{ "name": "stile (83)"},
{ "name": "upper sash (159)",
"children": [
{ "name": "muntin (109)"},
{ "name": "pane (279)"},
{ "name": "rail (86)"},
{ "name": "stile (114)"}
]
},
{ "name": "window (13)",
"children": [
{ "name": "muntin (10)"},
{ "name": "pane (33)"}
]
}
]
},
{ "name": "work surface (1204)"}
]
}
]

top_level_children = treeData[0]['children']

top_level_longnames = [child['name'] for child in top_level_children]

top_level_synset_strs = [name[:name.index('(')].strip()
                         for name in top_level_longnames]

top_level_synsets = [list(map(lambda x:x.strip(),s.split(',')))
                     for s in top_level_synset_strs]



def _ade20k_to_coco():
    from coco import coco
    mapping = {}
    for idx,adeitem in enumerate(top_level_synsets):
        found_match=False
        for jdx,cocoitem in enumerate(coco):
            for adek_syn in adeitem:
                for adek_syn_word in adek_syn.split():
                    if cocoitem == adek_syn_word:
                        #note this fails for e.g. gravy boat
                        mapping[idx]=jdx
                        found_match=True
        if not found_match:
            mapping[idx]=0
    return mapping

ade20k_to_coco = _ade20k_to_coco()

#from detector_foo import coco
#for adx,cdx in ade20k_to_coco().items():
#    print(top_level_synsets[adx],':',coco[cdx])
