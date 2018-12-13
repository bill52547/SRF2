#!/usr/bin/env python
# encoding: utf-8
'''
@author: Minghao Guo
@license: MIT
@contact: mh.guo0111@gmail.com
@software: personal
@file: unit_test_generator.py
@date: 12/5/2018
@desc: for personal usage
'''

import os
import os.path

path = '../srf2'

BRANCH = '├─'
LAST_BRANCH = '└─'
TAB = '│  '
EMPTY_TAB = '   '


def get_dir_list(path, placeholder = ''):
    folder_list = [folder for folder in os.listdir(path) if
                   os.path.isdir(os.path.join(path, folder))]
    file_list = [file for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))]
    result = ''
    for folder in folder_list[:-1]:
        if folder == 'test':
            continue
        result += placeholder + BRANCH + folder + '\n'
        result += get_dir_list(os.path.join(path, folder), placeholder + TAB)
    if folder_list:
        result += placeholder + (BRANCH if file_list else LAST_BRANCH) + folder_list[-1] + '\n'
        result += get_dir_list(os.path.join(path, folder_list[-1]),
                               placeholder + (TAB if file_list else EMPTY_TAB))
    for file in file_list[:-1]:
        result += placeholder + BRANCH + file + '\n'
    if file_list:
        result += placeholder + LAST_BRANCH + file_list[-1] + '\n'
    return result


def get_dir_list_tree(path, py_only = False, exceptation = ('test', '__init__.py',
                                                            'unit_test_generator.py')):
    folder_list = [folder for folder in os.listdir(path) if
                   os.path.isdir(os.path.join(path, folder))]
    file_list = [file for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))]
    result = {}
    for folder in folder_list:
        if folder in exceptation:
            continue
        subtree = get_dir_list_tree(os.path.join(path, folder), py_only, exceptation)
        if subtree:
            result[folder] = subtree

    for file in file_list:
        if file in exceptation:
            continue
        if py_only and not file.endswith('.py'):
            continue
        result[file] = os.path.join(path, file)
    return result


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        print('Directory', path, 'exist.')


def mknod(path):
    if not os.path.exists(path):
        os.mknod(path)
    else:
        print('File', path, 'exist.')


def create_dir_and_file(target, tree):
    mkdir(target)
    result1 = []
    result2 = []
    for key, value in tree.items():
        if isinstance(value, dict):
            res1, res2 = create_dir_and_file(os.path.join(target, key), value)
            result1 += res1
            result2 += res2
        if isinstance(value, str):
            mknod(os.path.join(target, 'tmp_test_' + key))
            result1 += [value]
            result2 += [os.getcwd() + '/' + os.path.join(target, 'tmp_test_' + key)]

    return result1, result2


def read_and_write_file(fin_path, fout_path):
    fin = open(fin_path, 'r')
    list_of_all_the_lines = fin.readlines()
    fin.close()

    fout = open(fout_path, 'w')
    for line in list_of_all_the_lines:
        if len(line.strip()) == 0:
            continue
        numspace = line.split(':')[0].count('    ')
        line = line.replace('    ', '')
        if len(line.split(' ')) < 2:
            continue
        type, name = line.split(' ')[0], line.split(' ')[1]
        name = name.split('(')[0]
        # print(type, name)

        if type == 'class':
            fout.writelines('class Test_' + name + ':\n')
        if type == 'def':
            name = name.replace('__', '')
            if numspace == 0:
                fout.writelines(numspace * '    ' + 'def test_' + name + '():\n')
            else:
                fout.writelines(numspace * '    ' + 'def test_' + name + '(self):\n')
            fout.writelines((numspace + 1) * '    ' + 'pass\n')
            fout.writelines('\n')


def create_unittests(path, target = 'auto_test'):
    file_tree = get_dir_list_tree(path, True)
    result1, result2 = create_dir_and_file(target, file_tree)

    for re1, re2 in zip(result1, result2):
        read_and_write_file(re1, re2)


if __name__ == "__main__":
    create_unittests('.', 'tmp_unit_test_templates')
