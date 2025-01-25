# -*- coding: utf-8 -*-
"""
@Project ：2023G 
@File ：format.py
@Author ：伍陆柒
@Desc ：
@Date ：2023/3/16 22:00 
"""

import pandas as pd
import csv
import numpy as np
import seaborn as sns
from scipy.stats import linregress
import datetime
import time

def write_header(filepath, fieldnames):
    with open(filepath, "a", encoding='utf-8', newline='') as fs:
        fieldnames = fieldnames
        writer = csv.DictWriter(fs, fieldnames=fieldnames)
        writer.writeheader()


def dataloader(file_path):
    df = pd.read_csv(file_path, encoding="gbk")
    df = df.fillna(method='ffill')
    # df = df.dropna(axis=0, how='any', inplace=True)
    return df


def format_2018(data):
    filepath = './2018.csv'

    fieldnames = ['no', 'gender', 'age', 'height', 'weight', 'bmi', 'griffith',
                  'thermal sensation', 'thermal accept', 'air sense',
                  'season', 'date', 'time', 'ta', 'hr']

    write_header(filepath, fieldnames)

    griffith = []

    for i in range(1, 7):
        data_person = data.loc[data["实验人员编号"] == i][['温度（℃）', '热感觉（-3~3）']]
        temp = data_person['温度（℃）']
        pmv = data_person['热感觉（-3~3）']
        scatter = sns.scatterplot(x=pmv, y=temp)
        scatter.set_xlabel('pmv')
        scatter.set_ylabel('temp')
        statistics = linregress(pmv, temp)
        griffith.append(statistics.slope)
    print(griffith)

    with open(filepath, "a", encoding='utf-8', newline='') as fs:
        for i in range(data.shape[0]):
            no = data.iloc[i, 0]
            age = data.iloc[i, 1]
            height = data.iloc[i, 2]
            if height < 170:
                gender = 0
            else:
                gender = 1
            weight = data.iloc[i, 3]
            bmi = round(data.iloc[i, 4], 2)
            season = 'summer'
            date = data.iloc[i, 5]
            time = data.iloc[i, 6]
            print(type(date))
            ta = data.iloc[i, 7]
            hr = data.iloc[i, 8]
            thermal_accept = data.iloc[i, 9]
            thermal_sensation = data.iloc[i, 10]
            air_sense = data.iloc[i, 11]
            grif = round(griffith[no - 1], 2)

            datalist = [no, gender, age, height, weight, bmi, grif,
                        thermal_sensation, thermal_accept, air_sense,
                        season, date, time, ta, hr]
            print(datalist)
            csv_write = csv.writer(fs)
            csv_write.writerow(datalist)


def format_2019_winter(data):
    filepath = './2019_winter.csv'

    fieldnames = ['no', 'gender', 'age', 'height', 'weight', 'bmi', 'griffith',
                  'thermal sensation', 'thermal preference', 'thermal acceptance',
                  'room',
                  'co2', 'air_speed',
                  'season', 'date', 'time', 'ta', 'hr']

    write_header(filepath, fieldnames)

    griffith = []

    for i in range(1, 11):
        data_person = data.loc[data["NO"] == i]
        temp = data_person['temp（℃）']
        pmv = data_person['thermal sensation(-3~3)']
        scatter = sns.scatterplot(x=pmv, y=temp)
        scatter.set_xlabel('pmv')
        scatter.set_ylabel('temp')
        statistics = linregress(pmv, temp)
        griffith.append(statistics.slope)
    print(len(griffith))

    with open(filepath, "a", encoding='utf-8', newline='') as fs:
        for i in range(data.shape[0]):
            print(data.iloc[i, 0])
            no = data.iloc[i, 0].astype(int)
            age = data.iloc[i, 1]
            gender = data.iloc[i, 2]
            height = data.iloc[i, 3]
            weight = data.iloc[i, 4]
            bmi = round(data.iloc[i, 5], 2)
            room = data.iloc[i, 6]
            date = data.iloc[i, 7]
            time = data.iloc[i, 8]
            ta = data.iloc[i, 9]
            hr = data.iloc[i, 10]
            co2 = data.iloc[i, 11]
            air_speed = data.iloc[i, 12]
            thermal_sensation = data.iloc[i, 13]
            thermal_accept = data.iloc[i, 14]
            thermal_preference = data.iloc[i, 15]
            grif = round(griffith[no - 1], 2)
            season = 'winter'

            datalist = [no, gender, age, height, weight, bmi, grif,
                        thermal_sensation, thermal_preference, thermal_accept,
                        room,
                        co2, air_speed, season, date, time, ta, hr]
            print(datalist)
            csv_write = csv.writer(fs)
            csv_write.writerow(datalist)


def format_2019_summer(data):
    filepath = './2019_summer.csv'

    fieldnames = ['no', 'gender', 'age', 'height', 'weight', 'bmi', 'griffith',
                  'sensitivity', 'environment',
                  'thermal sensation', 'thermal comfort', 'thermal preference',
                  'room', 'seat',
                  'season', 'date', 'time', 'ta', 'hr']

    write_header(filepath, fieldnames)

    number = np.array(data['number'].unique())

    griffith = []
    for i in range(0, len(number)):
        data_person = data.loc[data["number"] == number[i]]
        temp = data_person['temperature']
        pmv = data_person['thermal_sensation']
        scatter = sns.scatterplot(x=pmv, y=temp)
        scatter.set_xlabel('pmv')
        scatter.set_ylabel('temp')
        statistics = linregress(pmv, temp)
        griffith.append(statistics.slope)

    with open(filepath, "a", encoding='utf-8', newline='') as fs:
        for i in range(data.shape[0]):
            num = data.iloc[i, 11]
            for n in range(0, len(number)):
                if num == number[n]:
                    no = n + 1
            room = data.iloc[i, 1]
            seat = data.iloc[i, 2]
            thermal_sensation = data.iloc[i, 3]
            thermal_comfort = data.iloc[i, 4]
            thermal_preference = data.iloc[i, 5]
            age = data.iloc[i, 6]
            height = data.iloc[i, 7]
            weight = data.iloc[i, 8]
            bmi = round((weight / (height ** 2)), 2)
            sensation = data.iloc[i, 9] - 1  # environment
            sensitive = data.iloc[i, 10]
            if sensitive >= 3:
                sensitivity = 2
            else:
                sensitivity = sensitive
            gender = data.iloc[i, 12]
            t = data.iloc[i, 13]
            date = t.split(' ')[0]
            time = t.split(' ')[1]
            ta = data.iloc[i, 14]
            hr = data.iloc[i, 15]
            device_time = data.iloc[i, 16]
            grif = round(griffith[no - 1], 2)
            season = 'summer'

            datalist = [no, gender, age, height, weight, bmi, grif,
                        sensitivity, sensation,
                        thermal_sensation, thermal_comfort, thermal_preference,
                        room, seat,
                        season, date, time, ta, hr]
            print(datalist)
            # csv_write = csv.writer(fs)
            # csv_write.writerow(datalist)


def format_2021(data):
    filepath = './2021.csv'

    fieldnames = ['no', 'gender', 'age', 'height', 'weight', 'bmi', 'griffith',
                  'sensitivity', 'preference', 'environment',
                  'thermal sensation', 'thermal comfort', 'thermal preference',
                  'season', 'date', 'time', 'room', 'ta', 'hr']

    write_header(filepath, fieldnames)

    griffith = []
    for i in range(57):
        data_person = data.loc[data["no"] == i + 1]

        data_person = data_person[['thermal sensation', 'ta']]

        temp = data_person['ta']

        pmv = data_person['thermal sensation']

        scatter = sns.scatterplot(x=pmv, y=temp)

        scatter.set_xlabel('pmv')

        scatter.set_ylabel('temp')

        statistics = linregress(pmv, temp)

        griffith.append(statistics.slope)

        print(str(i + 1) + '号温度与PMV的斜率为' + str(statistics.slope))

    for i in range(len(data)):
        with open(filepath, "a", encoding='utf-8', newline='') as fs:
            no = data.iloc[i, 0]
            gender = data.iloc[i, 1]
            age = data.iloc[i, 2]
            height = data.iloc[i, 3]
            weight = data.iloc[i, 4]
            bmi = round(data.iloc[i, 5], 2)
            preference = data.iloc[i, 6]
            sensitivity = data.iloc[i, 7]
            environment = data.iloc[i, 8]
            grif = data.iloc[i, 9]
            thermal_sensation = data.iloc[i, 10]
            thermal_comfort = data.iloc[i, 11]
            thermal_preference = data.iloc[i, 12]
            season = data.iloc[i, 13]

            date = data.iloc[i, 14]
            time = data.iloc[i, 15]
            room = data.iloc[i, 16]

            temp = data.iloc[i, 17]
            humid = data.iloc[i, 18]

            datalist = [no, gender, age, height, weight, bmi, grif,
                        sensitivity, preference, environment,
                        thermal_sensation, thermal_comfort, thermal_preference,
                        season, date, time, room, temp, humid]
            print(datalist)
            csv_write = csv.writer(fs)
            csv_write.writerow(datalist)


if __name__ == "__main__":
    # data_2018 = dataloader('../OriDataSet/2018_summer.csv')
    # format_2018(data_2018)
    #
    # data_2019_summer = dataloader('../OriDataSet/2019_summer.csv')
    # format_2019_summer(data_2019_summer)
    #
    # data_2019_winter = dataloader('../OriDataSet/2019_winter.csv')
    # format_2019_winter(data_2019_winter)

    data_2021 = dataloader('../OriDataSet/2021.csv')
    format_2021(data_2021)