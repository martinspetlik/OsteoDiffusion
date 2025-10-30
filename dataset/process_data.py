import os
import numpy as np
import pyvista as pv
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


typ_kosti = 'L4'

if typ_kosti == "L4":
    data_dir = "/data/L4"
    

age_data_path = os.path.join(data_dir, 'age.npy')
sex_data_path = os.path.join(data_dir, 'sex.npy')
density_data_path = os.path.join(data_dir, 'rho.npy')
u_data_path = os.path.join(data_dir, 'u.npy')
template_data_path = os.path.join(data_dir, 'template.vtk')


age = np.load(age_data_path)
sex = np.load(sex_data_path)   # 0 - zenska, 1 - chlap
density = np.load(density_data_path)   # radky:pacienti; sloupce - bod v kosti
u = np.load(u_data_path)
template = pv.read(template_data_path) # geometrie kosti



n_points = template.n_points
n_cells = template.n_cells


print("cell 1", template.get_cell(1))


print("template n points: {}, n cells: {}".format(n_points, n_cells))

print("template type ", type(template))



#template.plot()

print("age shape ", age.shape)
print("sex shape ", sex.shape)
print("density shape ", density.shape)
print("u shape ", u.shape)

print("u[0] data ", u[0])

plt.hist(density[1, ...], label="density distribution") # density ve vsech bodech prvniho pacienta
plt.show()


print("Pocet pacientu={} | pocet bodu={}".format(density.shape[0], density.shape[0]))
print('Rozmezi ageu: {} az {} let'.format(age.min(), age.max()))


plt.hist(density[0, :]) # density ve vsech bodech prvniho pacienta
plt.show()

template['stredni_hodnota_vsichni'] = np.mean(density, axis=0)
template['odchylka_vsichni'] = np.std(density, axis=0)


# Filtrovani podle sex
density_zeny = density[sex==0, :]
density_muzi = density[sex==1, :]
template['stredni_hodnota_zeny'] = np.mean(density_zeny, axis=0)
template['odchylka_zeny'] = np.std(density_zeny, axis=0)
template['stredni_hodnota_muzi'] = np.mean(density_muzi, axis=0)
template['odchylka_muzi'] = np.std(density_muzi, axis=0)


# Filtrovani podle ageu
do_ageu = 40
age_zeny = age[sex==0]
density_zeny_do40 = density_zeny[age_zeny < do_ageu]
age_muzi = age[sex==1]
density_muzi_do40 = density_muzi[age_muzi < do_ageu]

template['stredni_hodnota_zeny_do40'] = np.mean(density_zeny_do40, axis=0)
template['odchylka_zeny_do40'] = np.std(density_zeny_do40, axis=0)
template['stredni_hodnota_muzi_do40'] = np.mean(density_muzi_do40, axis=0)
template['odchylka_muzi_do40'] = np.std(density_muzi_do40, axis=0)
template.save('statistika.vtk')



template.plot()



