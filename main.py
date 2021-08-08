import os
from os import path
import numpy as np
import scipy.signal
import scipy.io
import segyio
from mayavi import mlab


def main():


    dutch = Cube('Dutch Government_F3_entire_8bit seismic.segy', 'output_cubes')
    dutch.calc_analytic()
    dutch.calc_inst_freq()
    dutch.calc_inst_wavenumb_kx()
    dutch.calc_inst_wavenumb_ky()
    dutch.calc_dip()


# Como saída visual é usada a plataforma mayavi para exibir um cubo 3D interativo
    data = dutch.seismic_cube
    source = mlab.pipeline.scalar_field(data)
    source.spacing = [1, 1, -1]
    for axis in ['x', 'z']:
        plane = mlab.pipeline.image_plane_widget(source,
                                                 plane_orientation='{}_axes'.format(axis),
                                                 slice_index=100, colormap='gray')
        plane.module_manager.scalar_lut_manager.reverse_lut = True
    mlab.outline()
    mlab.show()


class Cube:

    # Definições básicas do cubo sísmico a ser processado, incluindo nome do arquivo em disco, número de inlines, crosslines
    # e timeslices. As constantes DIL, DXL e DTS representam as granularidade de distâncias e tempo, respectivamente eixos
    # x, y e z
    # Autor: Andre Grigorio
    def __init__(self, cube_filename, output_directory):
        seismic_cube_file = segyio.open(cube_filename)
        self.seismic_cube = segyio.tools.cube(seismic_cube_file)
        self.ILINE = seismic_cube_file.ilines.size
        self.XLINE = seismic_cube_file.xlines.size
        self.TSLICE = seismic_cube_file.samples.size
        self.DIL = 25 / self.ILINE
        self.DXL = 25 / self.XLINE
        self.DTS = segyio.tools.dt(seismic_cube_file) / 1000000

        if not os.path.isdir(output_directory):
            os.mkdir(output_directory)
        self.cubo_anal_file = os.path.join(output_directory, 'cubo_anal.npy')
        self.cubo_inst_frequ_file = os.path.join(output_directory, 'cubo_inst_frequ.npy')
        self.cubo_inst_wavenumb_kx_file = os.path.join(output_directory, 'cubo_inst_wavenumb_kx.npy')
        self.cubo_inst_wavenumb_ky_file = os.path.join(output_directory, 'cubo_inst_wavenumb_ky.npy')
        self.cubo_dip_mag_file = os.path.join(output_directory, 'cubo_dip_mag.npy')
        self.cubo_dip_az_file = os.path.join(output_directory, 'cubo_dip_az.npy')
        self.cubo_inst_dip_p_file = os.path.join(output_directory, 'cubo_inst_dip_p.npy')
        self.cubo_inst_dip_q_file = os.path.join(output_directory, 'cubo_inst_dip_q.npy')

    # A partir do cubo sísmico é obtido o cubo analítico, com a componente em quadratura do sinal, usando-se a transformada
    # de Hilbert. A transformada é executada unidimensionalmente, ao longo do eixo de tempo, para cada par inline e
    # crossline
    # Autor: Andre Grigorio
    def calc_analytic(self):
        if not path.exists(self.cubo_anal_file):
            self.cubo_anal = scipy.signal.hilbert(self.seismic_cube[:, :, :self.TSLICE])
            np.save(self.cubo_anal_file, self.cubo_anal)
        else:
            self.cubo_anal = np.load(self.cubo_anal_file)

    # A partir do cubo analítico é extraída a informação de fase do sinal e subsequentemente a frequência instantânea
    # é calculada pela diferenciação ao longo do eixo de tempo. Como a diferenciação usa o método de forward difference,
    # a última matriz do cubo é uma réplica da penúltima para que se mantenha a mesma ordem do objeto tridimensional
    # Autor: Andre Grigorio
    def calc_inst_freq(self):
        if not path.exists(self.cubo_inst_frequ_file):
            self.cubo_inst_frequ = np.zeros((self.ILINE, self.XLINE, self.TSLICE))
            self.cubo_inst_frequ[:, :, :-1] = np.diff(np.angle(self.cubo_anal))
            self.cubo_inst_frequ[:, :, -1] = self.cubo_inst_frequ[:, :, -2]
            np.save(self.cubo_inst_frequ_file, self.cubo_inst_frequ)
        else:
            self.cubo_inst_frequ = np.load(self.cubo_inst_frequ_file)

    # Similarmente ao procedimento anterior, são calculados os cubos de número de onda instantâneos kx e ky,
    # diferenciando-se ao longo dos eixos x e y, respectivamente
    # Autor: Andre Grigorio
    def calc_inst_wavenumb_kx(self):
        if not path.exists(self.cubo_inst_wavenumb_kx_file):
            self.cubo_inst_wavenumb_kx = np.zeros((self.ILINE, self.XLINE, self.TSLICE))
            self.cubo_inst_wavenumb_kx[:-1, :, :] = np.diff(np.angle(self.cubo_anal), axis=-3)
            self.cubo_inst_wavenumb_kx[-1, :, :] = self.cubo_inst_wavenumb_kx[-2, :, :]
            np.save(self.cubo_inst_wavenumb_kx_file, self.cubo_inst_wavenumb_kx)
        else:
            self.cubo_inst_wavenumb_kx = np.load(self.cubo_inst_wavenumb_kx_file)

    def calc_inst_wavenumb_ky(self):
        if not path.exists(self.cubo_inst_wavenumb_ky_file):
            self.cubo_inst_wavenumb_ky = np.zeros((self.ILINE, self.XLINE, self.TSLICE))
            self.cubo_inst_wavenumb_ky[:, :-1, :] = np.diff(np.angle(self.cubo_anal), axis=-2)
            self.cubo_inst_wavenumb_ky[:, -1, :] = self.cubo_inst_wavenumb_ky[:, -2, :]
            np.save(self.cubo_inst_wavenumb_ky_file, self.cubo_inst_wavenumb_ky)
        else:
            self.cubo_inst_wavenumb_ky = np.load(self.cubo_inst_wavenumb_ky_file)

    #  Com os cubos de frequência e número de ondas instantâneos, são calculados os cubos de dip aparentes, e, finalmente,
    #  os cubos de dip magnitude e dip azimuth
    # Autor: Andre Grigorio
    def calc_dip(self):
        if (not path.exists(self.cubo_dip_mag_file)) or (not path.exists(self.cubo_dip_az_file)) or (
        not path.exists(self.cubo_inst_dip_p_file)) or (not path.exists(self.cubo_inst_dip_q_file)):
            self.cubo_inst_dip_p = self.cubo_inst_wavenumb_kx / self.cubo_inst_frequ
            self.cubo_inst_dip_q = self.cubo_inst_wavenumb_ky / self.cubo_inst_frequ
            self.cubo_dip_mag = np.sqrt(np.square(self.cubo_inst_dip_p) + np.square(self.cubo_inst_dip_q))
            self.cubo_dip_az = np.arctan2(self.cubo_inst_dip_q, self.cubo_inst_dip_p)
            np.save(self.cubo_inst_dip_p_file, self.cubo_inst_dip_p)
            np.save(self.cubo_inst_dip_q_file, self.cubo_inst_dip_q)
            np.save(self.cubo_dip_mag_file, self.cubo_dip_mag)
            np.save(self.cubo_dip_az_file, self.cubo_dip_az)
        else:
            self.cubo_inst_dip_p = np.load(self.cubo_inst_dip_p_file)
            self.cubo_inst_dip_q = np.load(self.cubo_inst_dip_q_file)
            self.cubo_dip_mag = np.load(self.cubo_dip_mag_file)
            self.cubo_dip_az = np.load(self.cubo_dip_az_file)

if __name__ == "__main__":
    main()