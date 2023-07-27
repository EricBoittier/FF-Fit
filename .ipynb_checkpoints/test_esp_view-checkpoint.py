from ff_energy import *
CMS = load_all_theory_and_elec()
for i, cms in enumerate(CMS):
    print(i, cms)
i = int(input("?"))
# i = 14
esp_view_jobs([CMS[i]])
