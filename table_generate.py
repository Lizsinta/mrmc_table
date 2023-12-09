import os
from shutil import copyfile
from sys import argv, exit
from time import sleep
from os import makedirs, path, popen
from math import acos, log10
from time import perf_counter as timer

import numpy as np
import win32process
import win32event
import pywintypes
import numba

from ui_table import Ui_MainWindow as Ui_MainWindow_Table
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QApplication, QMessageBox
from PyQt5.QtCore import QThread, pyqtSignal
import icon_rc


@numba.jit(nopython=True)
def ic(decimals: int, ssc: np.array([]), fsc: np.array([]), sd: np.array([]), td: np.array([]),
       length: np.array([]), angle: np.array([])) -> list:
    lsize = length.size
    asize = angle.size
    focus_index = []
    focus_l = pi / 9
    for i in range(lsize):
        size_i = i * asize * lsize
        for j in range(asize):
            size_j = j * lsize
            for k in range(lsize):
                t_size = size_i + size_j + k
                ssc[t_size][0] = length[k] * cos(angle[j]) + length[i]
                ssc[t_size][1] = length[k] * sin(angle[j])
                sd[t_size] = round(sqrt((ssc[t_size] ** 2).sum()), decimals)
                td[t_size] = (length[i] + length[k] + sd[t_size]) / 2
                if td[t_size] < 6 and sd[t_size] > 1.5:
                    if ssc[t_size][0] > 0:
                        if angle[j] <= focus_l:
                            focus_index.append(t_size)
                    elif ssc[t_size][0] < 0:
                        a_range_2 = cal_angle(np.array([fsc[t_size], 0, 0]), np.zeros(3),
                                              np.array([ssc[t_size][0], ssc[t_size][1], 0]))
                        if a_range_2 <= focus_l:
                            focus_index.append(t_size)
    return focus_index

@numba.jit(nopython=True)
def cal_angle(coor1, coor2, coor3):
    vect1 = coor2 - coor1
    vect2 = coor3 - coor2
    transvection = round(np.sum(vect1 * vect2), 6)
    module = round(sqrt(np.sum(vect1 ** 2) * np.sum(vect2 ** 2)), 6)
    return round(acos(transvection / module) / pi * 180, 2)


def run_feff(folder=''):
    while True:
        try:
            info = win32process.CreateProcess('%s\\feff.exe' % folder, '',
                                              None, None, 0, win32process.CREATE_NO_WINDOW,
                                              None, '%s' % folder, win32process.STARTUPINFO())
        except pywintypes.error:
            continue
        else:
            break
    win32event.WaitForSingleObject(info[0], 2000)


def feff_table_s(dist, file, atom1, atom2, label, single=True):
    source = file + r'\temp'
    table = r'\table2' if single else r'\table2s'
    if label == 1:
        path = file + table
    elif label == 2:
        path = file + table + '_2'
    else:
        path = file + table + '_3'
    count = 0
    container = np.zeros(100, dtype=int)
    for step in range(dist.size):
        container[count] = step
        count += 1
        if count == 100 or dist[step] == dist[-1]:
            with open(source + r'\paths.dat', 'r+') as f:
                f.seek(0)
                while True:
                    lines = f.readline()
                    if not lines or not lines.find('---') == -1:
                        break
                if single:
                    for i in range(count):
                        f.seek(f.tell())
                        f.write('  %d 2 1\nx\n' % i)
                        f.seek(f.tell())
                        f.write(' %.6f 0 0 %d \'%s\'\n 0 0 0 0 \'%s\'\n' % (dist[container[i]], label, atom2, atom1))
                else:
                    for i in range(count):
                        f.seek(f.tell())
                        f.write('  %d 4 1\nx\n' % i)
                        f.seek(f.tell())
                        f.write(' %.6f 0 0 %d \'%s\'\n 0 0 0 0 \'%s\'\n %.6f 0 0 %d \'%s\'\n 0 0 0 0 \'%s\'\n'
                                % (dist[container[i]], label, atom2, atom1, dist[container[i]], label, atom2, atom1))
                f.truncate()
            run_feff(source)
            for i in range(count):
                name = r'\feff000%d.dat' % i if i < 10 else r'\feff00%d.dat' % i
                copyfile(source + name, path + r'\feff%d.dat' % container[i])
            count = 0


def feff_table_m(coor1, coor2, index, file, atom1, atom2, atom3, double=True):
    source = file + r'\temp'
    scatter_2nd = 1 if atom2 == atom3 else 2
    while True:
        try:
            with open(source + r'\paths.dat', 'r+') as f:
                f.seek(0)
                while True:
                    lines = f.readline()
                    if not lines or not lines.find('---') == -1:
                        break
                if double:
                    for i in range(index.size):
                        f.seek(f.tell())
                        f.write('  %d 3 1\nx\n' % i)
                        f.seek(f.tell())
                        f.write(' %.6f 0 0 1 \'%s\'\n %.6f %.6f 0 %d \'%s\'\n 0 0 0 0 \'%s\'\n'
                                % (coor1[index[i]], atom2, coor2[index[i]][0],
                                   coor2[index[i]][1], scatter_2nd, atom3, atom1))
                else:
                    for i in range(index.size):
                        f.seek(f.tell())
                        f.write('  %d 4 1\nx\n' % i)
                        f.seek(f.tell())
                        if coor2[index[i]][0] > 0:
                            f.write(' %.6f 0 0 1 \'%s\'\n %.6f %.6f 0 %d \'%s\'\n %.6f 0 0 1 \'%s\'\n 0 0 0 0 \'%s\'\n'
                                    % (coor1[index[i]], atom2, coor2[index[i]][0], coor2[index[i]][1], scatter_2nd,
                                       atom3, coor1[index[i]], atom2, atom1))
                        elif coor2[index[i]][0] < 0:
                            f.write(' %.6f 0 0 1 \'%s\'\n 0 0 0 0 \'%s\'\n %.6f %.6f 0 %d \'%s\'\n 0 0 0 0 \'%s\'\n'
                                    % (coor1[index[i]], atom2, atom1,
                                       coor2[index[i]][0], coor2[index[i]][1], scatter_2nd, atom3, atom1))
                f.truncate()
            break
        except PermissionError:
            sleep(0.5)
    run_feff(source)
    table = r'\table3' if double else r'\table4'
    for i in range(index.size):
        name = r'\feff000%d.dat' % i if i < 10 else r'\feff00%d.dat' % i
        copyfile(source + name, file + table + r'\feff%d.dat' % (index[i]))


class Worker(QThread):
    sig_statusbar = pyqtSignal(str, int)
    sig_currentLabel = pyqtSignal(int)
    sig_totalLabel = pyqtSignal(int)
    sig_progressBar = pyqtSignal(int)

    def __init__(self, parent=None):
        super(Worker, self).__init__(parent)
        self.l_head, self.l_tail = 1.5, 6.0
        self.step = 0.1
        self.element = np.array([])
        self.atom = ['Null', 'Null', 'Null', 'Null']
        self.atom_info = []
        self.r_init = np.array([])
        self.folder = ''
        self.inp_name = 'feff.inp'
        self.run_flag = False
        self.run_status = np.zeros(3, dtype=int)
        self.ms = False

        self.length = np.array([])
        self.angle = np.linspace(0, pi, 91)
        self.angle_degree = self.angle / pi * 180
        self.path_size = 0
        self.cal_amount = 0
        self.current_amount = 0
        self.p_counter = 0

        self.second_sc = np.array([])
        self.first_sc = np.array([])
        self.second_dist = np.array([])
        self.total_dist = np.array([])
        self.focus_index = np.array([], dtype=int)

    def read_inp(self, file_name):
        with open(file_name, 'r') as finp:
            while True:
                lines = finp.readline()
                if not lines:
                    return 'failed to read'
                if not lines.find(' POTENTIALS') == -1:
                    finp.readline()
                    temp = finp.readline()
                    self.atom[0] = temp.split()[2]
                    self.atom_info.append('   %s    %s' % (temp.split()[1], self.atom[0]))
                    atoms = np.array([])
                    while True:
                        lines = finp.readline()
                        if lines.isspace() or not lines.split()[0].isdecimal():
                            break
                        atoms = np.append(atoms, lines.split()[2])
                        self.atom_info.append('   %s    %s' % (lines.split()[1], atoms[-1]))
                    for i in range(atoms.size):
                        self.atom[i + 1] = atoms[i]
                    self.r_init = np.zeros(atoms.size + 1)
                    print(self.atom)
                    print(self.atom_info)
                    if atoms.size == 0:
                        return 'no scattering atoms found'
                    break

            while True:
                lines = finp.readline()
                if not lines or not lines.find('ATOMS') == -1:
                    break
            model_info = np.array([finp.readline().split()])
            while True:
                lines = finp.readline()
                if not lines or not lines.find('END') == -1:
                    break
                model_info = np.vstack((model_info, lines.split()))

            for i in range(model_info.shape[0]):
                if int(model_info[i][3]) == 0:
                    coor_a = np.array([float(model_info[i][0]), float(model_info[i][1]), float(model_info[i][2])])
                    break
            for i in range(model_info.shape[0]):
                dist = sqrt((float(model_info[i][0]) - coor_a[0]) ** 2 +
                            (float(model_info[i][1]) - coor_a[1]) ** 2 +
                            (float(model_info[i][2]) - coor_a[2]) ** 2)
                ipot = int(model_info[i][3])
                if self.r_init[ipot] == 0:
                    self.r_init[ipot] = dist
                else:
                    if self.r_init[ipot] > dist:
                        self.r_init[ipot] = dist
        self.inp_name = file_name.split('/')[-1]
        self.folder = file_name.split('/' + self.inp_name)[0]
        decimals = int(-log10(self.step))
        self.length = np.round(np.arange(self.l_head, self.l_tail + self.step, self.step), decimals)
        self.path_size = self.length.size ** 2 * self.angle.size


        self.cal_amount = (self.length.size + np.where(self.length <= 3)[0].size) * (self.r_init.size - 1)
        if self.ms:
            self.index_create()
            self.cal_amount += self.focus_index.size * (self.r_init.size - 1) * self.r_init.size
        return 'successfully read'

    def amount_change(self):
        decimals = int(-log10(self.step))
        self.length = np.round(np.arange(self.l_head, self.l_tail + self.step, self.step), decimals)
        self.path_size = self.length.size ** 2 * self.angle.size
        self.cal_amount = (self.length.size + np.where(self.length <= 3)[0].size) * (self.r_init.size - 1)
        if self.ms:
            self.index_create()
            self.cal_amount += self.focus_index.size * (self.r_init.size - 1) * self.r_init.size

    def file_init(self, name):
        self.sig_statusbar.emit('Inp initializing', 0)
        if not path.exists(self.folder + r'\temp'):
            makedirs(self.folder + r'\temp')
        popen('copy "%s" "%s"' % (self.folder + r'\%s' % self.inp_name, self.folder + r'\temp\feff.inp'))
        popen('copy "%s" "%s"' % (os.getcwd() + r'\%s' % name, self.folder + r'\temp\feff.exe'))
        sleep(0.5)
        with open(self.folder + r'\%s' % self.inp_name, 'r+') as f:
            while True:
                lines = f.readline()
                if not lines or not lines.find('ipot') == -1:
                    break
            while True:
                lines = f.readline()
                if lines.isspace() or not lines.split()[0].isdecimal():
                    break
        with open(self.folder + r'\temp\feff.inp', 'r+') as f:
            while True:
                position = f.tell()
                lines = f.readline()
                if not lines or not lines.find('CRITERIA') == -1:
                    break
            f.seek(position)
            data = ' CRITERIA     0    0'
            data = data.ljust(len(lines) - 1) + '\n'
            f.write(data)
            f.seek(0)
            while True:
                position = f.tell()
                lines = f.readline()
                if not lines or not lines.find('NLEG') == -1:
                    break
            f.seek(position)
            data = ' NLEG     4'
            data = data.ljust(len(lines) - 1) + '\n'
            f.write(data)
        with open(self.folder + r'\table.ini', 'w') as f:
            f.seek(0)
            f.seek(f.tell())
            f.write('Absorbing atom: %s\n' % self.atom[0])
            f.seek(f.tell())
            f.write('Scattering atom (1st): %s\n' % self.atom[1])
            f.seek(f.tell())
            f.write('Scattering atom (2nd): %s\n' % self.atom[2])
            f.seek(f.tell())
            f.write('Scattering atom (3rd): %s\n' % self.atom[3])
            f.seek(f.tell())
            f.write('Range: %.2f %.2f\n' % (self.l_head, self.l_tail))
            f.seek(f.tell())
            f.write('Step: %.2f\n' % self.step)
        self.sig_statusbar.emit('finished', 3000)

    def index_create(self):
        decimals = int(-log10(self.step))
        self.second_sc = np.zeros((self.path_size, 2))
        self.first_sc = np.broadcast_to(self.length, (self.angle.size * self.length.size, self.length.size)).T.flatten()
        self.second_dist = np.zeros(self.path_size)
        self.total_dist = np.zeros(self.path_size)
        start = timer()
        fi = ic(decimals, self.second_sc, self.first_sc, self.second_dist, self.total_dist,
                                         self.length, self.angle)
        self.focus_index = np.asarray(fi, dtype=int)
        print(timer() - start)

    def run(self):
        if self.run_status[0] == 0:
            for i in range(self.run_status[1], self.r_init.size - 1):
                if self.run_flag:
                    if self.run_status[2] == 0:
                        if not path.exists(self.folder + r'\table2_%d' % (self.run_status[1] + 1)):
                            makedirs(self.folder + r'\table2_%d' % (self.run_status[1] + 1))
                        if not path.exists(self.folder + r'\table2s_%d' % (self.run_status[1] + 1)):
                            makedirs(self.folder + r'\table2s_%d' % (self.run_status[1] + 1))
                    self.feff_table_s_full(self.run_status[1] + 1)
                    if self.run_flag:
                        self.run_status[2] = 0
                        self.run_status[1] += 1
                        print(self.run_status)
            if self.run_flag:
                self.run_status[0] += 1
                self.run_status[1] = 0
                print(self.run_status)

        if self.ms:
            if self.run_status[0] == 1:
                count = 0
                for j in range(1, self.r_init.size):
                    for i in range(1, j + 1):
                        if self.run_flag:
                            if count == self.run_status[1]:
                                if self.run_status[2] == 0:
                                    print(i, j, self.atom[i], self.atom[j])
                                    if not path.exists(self.folder + r'\table3_%d_%d' % (i, j)):
                                        makedirs(self.folder + r'\table3_%d_%d' % (i, j))
                                    if not path.exists(self.folder + r'\table4_%d_%d' % (i, j)):
                                        makedirs(self.folder + r'\table4_%d_%d' % (i, j))
                                self.feff_table_m_full(i, j)
                                if self.run_flag:
                                    self.run_status[2] = 0
                                    self.run_status[1] += 1
                                    print(self.run_status)
                            count += 1
                if self.run_flag:
                    self.run_status[1] = 0
                    self.run_status[0] = 0
                    print(self.run_status)
        else:
            self.run_status[1] = 0
            self.run_status[0] = 0
            print(self.run_status)
        self.sig_statusbar.emit('finished', 3000)

    def potential_cal(self, label, pot=True):
        with open(self.folder + r'\temp\feff.inp', 'r+') as f:
            f.seek(0)
            while True:
                lines = f.readline()
                if not lines or not lines.find('ff2chi') == -1:
                    break
            line_c = f.tell()
            length_c = len(f.readline())
            line_p = f.tell()
            length_p = len(f.readline())
            data = ' CONTROL   1     1     1     1     1     1'
            data = data.ljust(length_c - 1) + '\n'
            f.seek(line_c)
            f.write(data)
            data = ' PRINT     0     0     0     0     0     0'
            data = data.ljust(length_p - 1) + '\n'
            f.seek(line_p)
            f.write(data)
            f.seek(0)
            while True:
                lines = f.readline()
                if not lines or not lines.find('POTENTIALS') == -1:
                    break
            f.readline()
            line_a = f.tell()
            length_a = len(f.readline())
            f.seek(f.tell())
            data = '       %d%s' % (0, self.atom_info[0])
            data = data.ljust(length_a - 1) + '\n'
            f.seek(line_a)
            f.write(data)
            for i in range(len(label)):
                line_a = f.tell()
                length_a = len(f.readline())
                f.seek(f.tell())
                data = '       %d%s' % (i + 1, self.atom_info[label[i]])
                data = data.ljust(length_a - 1) + '\n'
                f.seek(line_a)
                f.write(data)
            while True:
                line_i = f.tell()
                data_i = f.readline()
                length_i = len(data_i)
                temp = data_i.split()
                if len(temp) == 0:
                    continue
                if not temp[0].isdecimal():
                    break
                f.seek(line_i)
                f.write(' '.ljust(length_i - 1) + '\n')
            f.seek(0)
            f.readline()
            f.readline()
            while True:
                lines = f.readline()
                if not lines or not lines.find('ATOMS') == -1:
                    break
            f.seek(f.tell())
            f.write('   0.00000     0.00000    0.00000     0   %s\n' % self.atom[0])
            for i in range(len(label)):
                line_a = f.tell()
                length_a = len(f.readline())
                coor = np.zeros(3)
                coor[i] += self.r_init[label[i]]
                data = '   %.5f     %.5f    %.5f     %d   %s' % (coor[0], coor[1], coor[2], i + 1, self.atom[label[i]])
                data = data.ljust(length_a - 1) if length_a > len(data) else data
                f.seek(line_a)
                f.write(data + '\n')
            f.seek(f.tell())
            f.write('END\n')
            f.truncate()
        try:
            run_feff(folder=self.folder + r'\temp')
        except PermissionError:
            sleep(0.5)
            run_feff(folder=self.folder + r'\temp')
        sleep(0.5)
        with open(self.folder + r'\temp\feff.inp', 'r+') as f:
            f.seek(0)
            while True:
                lines = f.readline()
                if not lines or not lines.find('ff2chi') == -1:
                    break
            line_c = f.tell()
            length_c = len(f.readline())
            line_p = f.tell()
            length_p = len(f.readline())
            if not pot:
                f.seek(line_c)
                data = ' CONTROL   0     1     1     1     1     1'
                data = data.ljust(length_c - 1) + '\n'
                f.write(data)
            f.seek(line_p)
            data = ' PRINT     0     0     0     0     0     3'
            data = data.ljust(length_p - 1) + '\n'
            f.write(data)

    def feff_table_s_full(self, label):
        source = self.folder + r'\temp'
        table_single = self.folder + r'\table2_%d' % label
        table_commute = self.folder + r'\table2s_%d' % label
        if self.run_status[2] == 0:
            self.potential_cal([label], pot=True)
        for step in range(self.run_status[2], self.length.size):
            if self.run_flag:
                with open(source + r'\feff.inp', 'r+') as f:
                    f.seek(0)
                    f.readline()
                    f.readline()
                    while True:
                        lines = f.readline()
                        if not lines or not lines[:10].find('ATOMS') == -1:
                            break
                    f.readline()
                    f.seek(f.tell())
                    f.write(' %.6f 0 0 1 %s\n' % (self.length[step], self.atom[label]))
                    f.write('END')
                    f.truncate()
                run_feff(source)
                copyfile(source + r'\feff0001.dat', table_single + r'\feff%d.dat' % step)
                if self.length[step] <= 3:
                    copyfile(source + r'\feff0002.dat', table_commute + r'\feff%d.dat' % step)
                    self.current_amount += 2
                else:
                    self.current_amount += 1
                self.sig_currentLabel.emit(self.current_amount)
                self.sig_progressBar.emit(int(self.current_amount / self.cal_amount * 100))
                self.run_status[2] += 1
            else:
                return

    def feff_table_m_full(self, sc1, sc2):
        source = self.folder + r'\temp'
        scatter_2nd = 1 if self.atom[sc1] == self.atom[sc2] else 2
        table_double = self.folder + r'\table3_%d_%d' % (sc1, sc2)
        table_triple = self.folder + r'\table4_%d_%d' % (sc1, sc2)
        if self.run_status[2] == 0:
            self.potential_cal([sc1, sc2], pot=False)
        for i in range(self.run_status[2], self.focus_index.size):
            if self.run_flag:
                if self.total_dist[self.focus_index[i]] < 6:
                    with open(source + r'\feff.inp', 'r+') as f:
                        f.seek(0)
                        f.readline()
                        f.readline()
                        while True:
                            lines = f.readline()
                            if not lines or not lines.find('ATOMS') == -1:
                                break
                        f.readline()
                        f.seek(f.tell())
                        f.write(' %.6f 0 0 1 %s\n %.6f %.6f 0 %d %s\n'
                                % (self.first_sc[self.focus_index[i]], self.atom[sc1],
                                   self.second_sc[self.focus_index[i]][0],
                                   self.second_sc[self.focus_index[i]][1], scatter_2nd, self.atom[sc2]))
                        f.write('END\n')
                        f.truncate()
                    run_feff(source)
                    tri_valid = False
                    with open(source + r'\paths.dat', 'r') as f:
                        f.readline()
                        f.readline()
                        while True:
                            lines = f.readline()
                            if not lines:
                                break
                            temp = lines.split()
                            f.readline()
                            seq = np.array([])
                            nleg = int(temp[1])
                            for _ in range(nleg):
                                seq = np.append(seq, int(f.readline().split()[3]))
                            if nleg == 3:
                                copyfile(source + r'\feff000%d.dat' % int(temp[0]), table_double + r'\feff%d.dat' % i)
                                self.current_amount += 1
                            elif nleg == 4:
                                if np.where(seq == np.array([1, 2, 1, 0]), True, False).all() \
                                        or np.where(seq == np.array([2, 1, 2, 0]), True, False).all():
                                    tri_valid = True
                                    copyfile(source + r'\feff000%d.dat' % int(temp[0]),
                                             table_triple + r'\feff%d.dat' % i)
                                    self.current_amount += 1
                                    break
                    if not tri_valid:
                        self.cal_amount -= 1
                        self.sig_totalLabel.emit(self.cal_amount)
                else:
                    self.cal_amount -= 2
                    self.sig_totalLabel.emit(self.cal_amount)
                self.sig_currentLabel.emit(self.current_amount)
                self.sig_progressBar.emit(int(self.current_amount / self.cal_amount * 100))
                self.run_status[2] += 1
            else:
                return


class MainWindow(QMainWindow, Ui_MainWindow_Table):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.setWindowTitle('Table generator')
        self.setWindowIcon(QIcon(':\\mRMC.ico'))
        self.setMinimumSize(0, 0)

        self.thread = Worker()

        self.startBox.setValue(self.thread.l_head)
        self.endBox.setValue(self.thread.l_tail)
        self.absorbLine.setText(self.thread.atom[0])
        self.scat1Line.setText(self.thread.atom[1])
        self.scat2Line.setText(self.thread.atom[2])

        self.thread.sig_statusbar.connect(self.statusbar_print)
        self.thread.sig_currentLabel.connect(self.currentLabel_change)
        self.thread.sig_totalLabel.connect(self.totalLabel_change)
        self.thread.sig_progressBar.connect(self.progressBar_change)

        self.startButton.setEnabled(False)
        self.pauseButton.setEnabled(False)
        self.endButton.setEnabled(False)
        self.actionRead_inp.triggered.connect(self.read_inp)
        self.startBox.valueChanged.connect(self.range_change)
        self.endBox.valueChanged.connect(self.range_change)
        self.stepBox.currentIndexChanged.connect(self.range_change)
        self.startButton.clicked.connect(self.generate_start)
        self.pauseButton.clicked.connect(self.generate_pause)
        self.endButton.clicked.connect(self.generate_stop)
        self.msBox.clicked.connect(self.ms_switch)

    def read_inp(self):
        file_name = QFileDialog.getOpenFileName(self, 'select inp file...', path.abspath('../..'), filter='*.inp')
        if file_name[0] == '':
            return
        self.statusbar.showMessage('Reading', 0)
        result = self.thread.read_inp(file_name[0])
        if result == 'successfully read':
            self.absorbLine.setText(self.thread.atom[0])
            self.scat1Line.setText(self.thread.atom[1])
            if not self.thread.atom[1] == self.thread.atom[2]:
                self.scat2Line.setText(self.thread.atom[2])
            else:
                self.scat2Line.setText('Null')
            if not self.thread.atom[2] == self.thread.atom[3]:
                self.scat3Line.setText(self.thread.atom[3])
            else:
                self.scat3Line.setText('Null')
            self.folderLabel.setText(self.thread.folder)
            self.totalLabel.display(self.thread.cal_amount)
            self.startButton.setEnabled(True)
            self.pauseButton.setEnabled(False)
            self.endButton.setEnabled(False)
        self.statusbar.showMessage(result, 3000)

    def generate_start(self):
        flist = np.asarray(os.listdir(os.getcwd()))
        flist = flist[np.where(np.char.find(flist, 'feff') >= 0)[0]]
        feff = np.where(np.char.find(flist, 'exe') >= 0)[0]
        if not feff.size > 0:
            QMessageBox.critical(self, 'Error',
                                 'feff8.exe not found.\nPlease put it on the same folder as this program')
            return
        feff = flist[feff[0]]
        print(feff)
        self.startButton.setEnabled(False)
        self.pauseButton.setEnabled(True)
        self.endButton.setEnabled(True)
        self.msBox.setEnabled(False)
        print('start', self.thread.run_status)
        if self.thread.run_status.sum() == 0:
            self.thread.file_init(feff)
        # self.index_create()
        self.statusbar.showMessage('Generation running', 0)
        self.thread.run_flag = True
        self.thread.start()

    def generate_pause(self):
        self.startButton.setEnabled(True)
        self.pauseButton.setEnabled(False)
        self.endButton.setEnabled(True)
        self.thread.run_flag = False
        if self.thread.isRunning():
            print('here')
            self.thread.wait()
            print('finish')
        print('pause', self.thread.run_status)
        self.statusbar.showMessage('Pause', 0)
        pass

    def generate_stop(self):
        self.startButton.setEnabled(True)
        self.pauseButton.setEnabled(False)
        self.endButton.setEnabled(False)
        self.msBox.setEnabled(True)
        self.thread.run_flag = False
        if self.thread.isRunning():
            self.thread.wait()
        self.thread.run_status[0] = 0
        self.thread.run_status[1] = 0
        self.thread.current_amount = 0
        single_amount = self.thread.length.size + np.where(self.thread.length <= 3)[0].size
        if not self.thread.atom[1] == self.thread.atom[2]:
            single_amount *= 2
        self.thread.cal_amount = single_amount + self.thread.focus_index.size * 2
        self.currentLabel.display(0)
        self.totalLabel.display(self.thread.cal_amount)
        self.progressBar.setValue(0)
        self.statusbar.showMessage('Stop', 3000)
        pass

    def range_change(self):
        signal = self.sender()
        if signal.hasFocus():
            self.startButton.setEnabled(False)
            if signal.objectName() == 'startBox':
                mini = self.startBox.value()
                if mini > self.thread.l_tail - self.thread.step:
                    self.thread.l_head = self.thread.l_tail - self.thread.step
                    self.startBox.setValue(self.thread.l_head)
                else:
                    self.thread.l_head = mini
            elif signal.objectName() == 'endBox':
                maxi = self.endBox.value()
                if maxi < self.thread.l_head + self.thread.step:
                    self.thread.l_tail = self.thread.l_head + self.thread.step
                    self.endBox.setValue(self.thread.l_tail)
                else:
                    self.thread.l_tail = maxi
            elif signal.objectName() == 'stepBox':
                self.thread.step = float(self.stepBox.currentText())
            if not self.thread.focus_index.size == 0:
                self.thread.amount_change()
                self.totalLabel.display(self.thread.cal_amount)
                self.startButton.setEnabled(True)
            print(self.thread.l_head, self.thread.l_tail)

    def ms_switch(self):
        if self.msBox.isChecked():
            self.thread.ms = True
            self.thread.cal_amount += self.thread.focus_index.size * \
                                      (self.thread.r_init.size - 1) * self.thread.r_init.size
        else:
            self.thread.ms = False
            self.thread.cal_amount -= self.thread.focus_index.size * \
                                      (self.thread.r_init.size - 1) * self.thread.r_init.size
        self.totalLabel_change(self.thread.cal_amount)

    def statusbar_print(self, message, time):
        self.statusbar.showMessage(message, time)

    def currentLabel_change(self, value):
        self.currentLabel.display(value)

    def totalLabel_change(self, value):
        self.totalLabel.display(value)

    def progressBar_change(self, value):
        self.progressBar.setValue(value)


if __name__ == '__main__':
    from math import pi, sin, cos, sqrt

    app = QApplication(argv)
    main = MainWindow()
    main.show()
    exit(app.exec_())
