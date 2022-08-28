classa = 'minor'
classb = 'moderate'
classc = 'severe'

type_dict = {classa: (0., 2e2),
             classb: (2e2, 5e5),
             classc: (5e5, 1e99),
             'dur_xmin': 5,
             'area_xmin': 30,
             'vox_xmin': 200,
             'area_alpha_mnx': (-3.5, -1.01),
             'vox_alpha_mnx': (-3.5, -1.01),
             }

type_dict_km = {
    classa: (0., 1e5),
    classb: (1e5, 1e8),
    classc: (1e8, 1e99),
             'dur_xmin': 5,
             'area_xmin': 1e4,
             'vox_xmin': 1e5,
             }
