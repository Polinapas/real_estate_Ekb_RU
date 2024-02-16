#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"></ul></div>

# In[71]:


get_ipython().system('pip install pandas')
get_ipython().system('pip install geopy')


# In[72]:


# импортируем необходимые библиотеки
import numpy as np
import pandas as pd
import os
import geopy.distance
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from geopy.distance import geodesic
from scipy.stats import kruskal
from scipy.stats import kruskal
from matplotlib import pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib as mpl
import seaborn as sns
import pylab


# In[73]:


ad = pd.read_csv('анализ данных.csv', sep=';') # читаем файл формата csv


# In[74]:


ad['price_per_metr'] = ad.price/ad.area # добавляем новую переменную - цену за квадратный метр жилья


# In[75]:


ad.head() # просмотр таблицы


# In[76]:


# Таблица 3. Пропущенные значения и размерность до установки ограничений (стр. 8)
print(ad.isna().sum()) # определяем количество пропущенных значений
print(f'Размерность: {ad.shape}') 


# In[77]:


# удаляем ненужные столбцы, в которых есть пропуски
ad.drop(columns=['postal_code', 'street_id', 'house_id', 'date'], inplace=True)


# In[78]:


print(ad.isna().sum()) # проверяем результат
print(f'Размерность: {ad.shape}')


# In[79]:


tym_ad = ad[ad.id_region == 72] # регион - тюмень
# добавляем ограничения по координатам, чтобы в выборку входила только тюмень, не область
tym_ad =tym_ad[tym_ad.geo_lat > 57]
tym_ad =tym_ad[tym_ad.geo_lat < 57.9]
tym_ad =tym_ad[tym_ad.geo_lon > 65]
tym_ad =tym_ad[tym_ad.geo_lon < 65.9]
tym_ad =tym_ad[tym_ad.building_type > 0] # убираем неизвестный тип здания


# In[80]:


data_before = tym_ad # создаем новую переменную для оценки ДО и ПОСЛЕ удаления выбросов и фильтрации
data_before.info()


# In[81]:


# до удаления выбросов
data_describe = tym_ad.drop(['building_type', 'object_type'], axis=1)


# In[82]:


tym_ad.drop(columns=['price', 'id_region', 'geo_lat', 'geo_lon' ], inplace=True) # удаляем ненужные столбцы с пустыми значениями


# In[83]:


# Статистические свойства количественных факторов до удаления выбросов
# до удаления выбросов
data_describe.describe()
# наше внимание привлекли аномально высокие значения в price, а также отрицательные значения в rooms, kitchen_area


# In[84]:


data_frame = pd.DataFrame(data=data_describe)
fig, axes = plt.subplots(1, figsize = (6.5, 6.5))
sns.set()
fig.suptitle('До удаления выбросов')

sns.scatterplot(x='price_per_metr', y='area', data=data_describe, alpha = 0.02, color='darkblue')
plt.xlabel('площадь в м^2')
plt.ylabel('цена за кв.м')


# In[85]:


fig, axes = plt.subplots(1, figsize = (6.5, 6.5)) # указали кол-во фигур на листе, размер фигуры
sns.set()
fig.suptitle('До удаления выбросов') # указали название 

sns.scatterplot(y='levels', x='price_per_metr', data=data_describe, alpha = 0.02, color='darkblue') # прописали параметры: переменную х, у, данные, блюр, цвет # добавили имя для оси ОУ
plt.xlabel('этажность дома') # добавили имя для оси ОХ
plt.ylabel('цена за кв.м')


# In[86]:


h = data_describe['price_per_metr'].hist()
fig = h.get_figure()
# заметны аномальные результаты


# In[87]:


# прописываем ограничения
tym_ad = ad[ad.id_region == 72] # регион - тюмень
tym_ad = tym_ad[tym_ad.rooms >= 1] # комнат - больше 1
tym_ad = tym_ad[tym_ad.kitchen_area > 1] # площадь кухни - больше 0
tym_ad =tym_ad[tym_ad.area > 1] # площадь всей недвижимости - больше 0
tym_ad =tym_ad[tym_ad.levels >= 0] # этажность здания - больше или равно 0
tym_ad =tym_ad[tym_ad.price_per_metr > 0]# цена за кв. м. - больше 0
tym_ad =tym_ad[tym_ad.building_type > 0] 
tym_ad =tym_ad[tym_ad.price_per_metr < 1000000] # устанавливаем границу максимальной стоимости
                                                # за кв.метр в Тюмени с запасом
tym_ad =tym_ad[tym_ad.price_per_metr > 1000] # устанавливаем границу минимальной стоимости
                                                # за кв.метр в Тюмени с запасом    
print(tym_ad.price_per_metr.min()) 
print(tym_ad.price_per_metr.max())
print(tym_ad.head(10))
print(tym_ad.shape)

# минимальная и максимальная стоимость кв.метра вызывает вопросы, однако она уже более реалистична


# In[88]:


# Таблица 4. Размерность до и после удаления выбросов (стр. 8)
# удаляем выбросы, используя правило 3х сигм
print(f'Размерность до удаления выбросов: {tym_ad.shape}')
mean = np.mean(tym_ad.price_per_metr)
std = np.std(tym_ad.price_per_metr)

# Определение границ, соответствующих правилу трёх сигм
lower_bound = mean - 3 * std
upper_bound = mean + 3 * std

# Фильтрация значений, находящихся вне границ
tym_ad = tym_ad[(tym_ad.price_per_metr >= lower_bound) & (tym_ad.price_per_metr <= upper_bound)]
print(tym_ad.head(10))
print(f'Размерность до удаления выбросов: {tym_ad.shape}')

print(tym_ad.price_per_metr.min()) 
print(tym_ad.price_per_metr.max())


# In[89]:


# считаем расстояние до центра
tym_centre = (57.152985, 65.541227) # нулевой километр 
def calculate_distance(row): 
    center_coords = (row['geo_lat'], row['geo_lon'])
    distance = geodesic(center_coords, tym_centre).kilometers
    return distance
tym_ad['distance'] = tym_ad.apply(calculate_distance, axis=1)


# In[90]:


# удаляем ненужные столбцы
tym_ad.drop(columns=['id_region', 'geo_lat', 'geo_lon', 'price'], inplace=True)


# In[91]:


tym_ad.info() # проверяем формат данных


# In[92]:


tym_ad


# In[93]:


# создаем новую переменную - только количественные данные
data_kol = tym_ad.drop(['building_type', 'object_type'], axis=1)
data_kol


# In[95]:


# Здесь будут значения оценок коэффициента корреляции Пирсона
C_P = pd.DataFrame([], index=data_kol.columns, columns=data_kol.columns)
# Здесь будут значения значимости оценок коэффициента корреляции Пирсона
P_P = pd.DataFrame([], index=data_kol.columns, columns=data_kol.columns)
# Здесь будут значения оценок коэффициента корреляции Спирмена
C_S = pd.DataFrame([], index=data_kol.columns, columns=data_kol.columns)
# Здесь будут значения значимости оценок коэффициента корреляции Спирмена
P_S = pd.DataFrame([], index=data_kol.columns, columns=data_kol.columns)
for x in data_kol.columns:
    for y in data_kol.columns:
        C_P.loc[x,y], P_P.loc[x,y] = pearsonr(data_kol[x], data_kol[y])
        C_S.loc[x,y], P_S.loc[x,y] = spearmanr(data_kol[x], data_kol[y])
# Сохраняем текстовый отчет на разные листы Excel файла
with pd.ExcelWriter('пирсон спирмен.xlsx', engine="openpyxl") as wrt:
  # Общая статистика
    data_kol.to_excel(wrt, sheet_name='stat')
# Корреляция Пирсона
    C_P.to_excel(wrt, sheet_name='Pearson')
    dr = C_P.shape[0] + 2
    P_P.to_excel(wrt, startrow=dr, sheet_name='Pearson') # Значимость
# Корреляция Спирмена
    C_S.to_excel(wrt, sheet_name='Spirmen')
    dr = C_S.shape[0] + 2
    P_S.to_excel(wrt, startrow=dr, sheet_name='Spirmen') # Значимость


# In[96]:


# подготовка к визуализации
C_P = C_P.rename(columns={'level': 'этаж'})
C_P = C_P.rename(columns={'levels': 'этажность здания'})
C_P = C_P.rename(columns={'rooms': 'кол-во комнат'})
C_P = C_P.rename(columns={'area': 'площадь'})
C_P = C_P.rename(columns={'kitchen_area': 'площадь кухни'})
C_P = C_P.rename(columns={'price_per_metr': 'цена за м^2'})
C_P = C_P.rename(columns={'distance': 'расстояние до центра'})
C_P


# In[97]:


# подготовка к визуализации
C_S = C_S.rename(columns={'level': 'этаж'})
C_S = C_S.rename(columns={'levels': 'этажность здания'})
C_S = C_S.rename(columns={'rooms': 'кол-во комнат'})
C_S = C_S.rename(columns={'area': 'площадь'})
C_S = C_S.rename(columns={'kitchen_area': 'площадь кухни'})
C_S = C_S.rename(columns={'price_per_metr': 'цена за м^2'})
C_S = C_S.rename(columns={'distance': 'расстояние до центра'})
C_S


# In[99]:


# Таблица 6. Критерий Крускала-Уоллиса для переменных (стр. 9)
# Создаем подвыборки
sel_1 = tym_ad['building_type']==1
x_1 = tym_ad.loc[sel_1, 'price_per_metr']
sel_2 = tym_ad['building_type']==2
x_2 = tym_ad.loc[sel_2, 'price_per_metr']
sel_3 = tym_ad['building_type']==3
x_3 = tym_ad.loc[sel_3, 'price_per_metr']
sel_4 = tym_ad['building_type']==4
x_4 = tym_ad.loc[sel_4, 'price_per_metr']
sel_5 = tym_ad['building_type']==5
x_5 = tym_ad.loc[sel_5, 'price_per_metr']
sel_6 = tym_ad['building_type']==6
x_6 = tym_ad.loc[sel_6, 'price_per_metr']

# Используем криетрий Крускала-Уоллиса
Price_sig = kruskal(x_1, x_2, x_3, x_4, x_5, x_6)
# Используем криетрий Крускала-Уоллиса

print('Критерий Крускала-Уоллиса для переменных \'цена за м^2\' и \'Тип здания\'')
print(Price_sig)
with open('tyme.txt', 'a') as fln:
    print('Критерий Крускала-Уоллиса для переменных \'цена за м^2\' и \'Тип здания\'',
          file=fln)
    print(Price_sig, file=fln)


# In[101]:


# Критерий Крускала-Уоллиса для переменных
# Создаем подвыборки
se_0 = tym_ad['object_type']==0
с_1 = tym_ad.loc[se_0, 'price_per_metr']
se_2 = tym_ad['object_type']==2
с_2 = tym_ad.loc[se_2, 'price_per_metr']


print('Критерий Крускала-Уоллиса для переменных \'цена за м^2\' и \'Тип жилья\'')
print(Price_sig)
with open('tyme.txt', 'a') as fln:
    print('Критерий Крускала-Уоллиса для переменных \'цена за м^2\' и \'Тип жилья \'',
          file=fln)
    print(Price_sig, file=fln)


# In[102]:


# Таблица 6. Критерий Крускала-Уоллиса для переменных (стр. 9)
tym_dvlp = tym_ad.astype({'price_per_metr':np.float64, 'building_type':'category', 
                      'object_type':'category'})
dvlp = tym_dvlp.copy()
crtx = pd.crosstab(dvlp['building_type'], dvlp['object_type'], margins=True)
# Даем имена переменным
crtx.columns.name = 'building_type'
crtx.index.name = 'object_type'
# Из уже готовой таблицы сопряженности
# Создаем объект sm.stats.Table для проведения анализа
# В объекте находятся все необходимые статистики и дополнительные методы
tabx = sm.stats.Table(crtx)
# Сохраняем полученные результаты

with pd.ExcelWriter('ad_tym-качественные.xlsx', engine="openpyxl") as wrt:
# Таблица сопряженности
    tabx.table_orig.to_excel(wrt, sheet_name='Тип здания-Тип жилья')
    dr = tabx.table_orig.shape[0] + 2 # Смещение по строкам
# Ожидаемые частоты при независимости
    tabx.fittedvalues.to_excel(wrt, sheet_name='Тип здания-Тип жилья', startrow=dr)
# Критерий хи квадрат для номинальных переменных
resx = tabx.test_nominal_association()
# Сохраняем результат в файле
with open('tyme.txt', 'a') as fln:
    print('Критерий HI^2 для переменных \'Тип здания\' и \'Тип жилья \'',
          file=fln)
    print(resx, file=fln)


# In[103]:


# Таблица 6. Критерий Крускала-Уоллиса для переменных (стр. 9)
# Рассчет Cramer V по формуле
nr = tabx.table_orig.shape[0]
nc = tabx.table_orig.shape[1]
N = tabx.table_orig.iloc[nr-1, nc-1]
hisq = resx.statistic
CrV = np.sqrt(hisq/(N*min((nr - 1, nc - 1))))
with open('tyme.txt', 'a') as fln:
    print('Статистика Cramer V для переменных \'Тип здания\' и \'Тип жилья \'',
          file=fln)
    print(CrV, file=fln)


# In[104]:


# подготовка к визуализации
dvlp['object_type'] = dvlp['object_type'].replace(0, 'Вторичка')
dvlp['object_type'] = dvlp['object_type'].replace(2, 'Новостройка')
dvlp['building_type'] = dvlp['building_type'].replace(1, 'Другое')
dvlp['building_type'] = dvlp['building_type'].replace(2, 'Панельный')
dvlp['building_type'] = dvlp['building_type'].replace(3, 'Монолитный')
dvlp['building_type'] = dvlp['building_type'].replace(4, 'Кирпичный')
dvlp['building_type'] = dvlp['building_type'].replace(5, 'Блочный')
dvlp['building_type'] = dvlp['building_type'].replace(6, 'Деревянный')
dvlp = dvlp.rename(columns={'object_type': 'Тип жилья'})
dvlp = dvlp.rename(columns={'building_type': 'Тип здания'})
dvlp


# In[105]:


data_before = data_before.astype({'price_per_metr':np.float64, 'building_type':'category', 
                      'object_type':'category'})
DB = data_before.copy()
crtxDB = pd.crosstab(DB['building_type'], DB['object_type'], margins=True)


# In[107]:


# Рис. 3. Столбчатые диаграммы – качественные переменные (стр. 8)
dfn = dvlp.select_dtypes(include=['O', "category"])
plt.figure(figsize=(15, 9)) # Дюймы
ftb_1 = pd.crosstab(dfn['Тип жилья'], 'Тип жилья')
ax=plt.subplot(2, 2, 1)
ftb_1.plot.bar(ax=ax, grid=True, legend=False,
               color='Blue')
ftb_2 = pd.crosstab(dfn['Тип здания'], 'Тип здания')
ax=plt.subplot(2, 2, 2)
ftb_2.plot.bar(ax=ax, grid=True, legend=False,
               color='Blue')
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.suptitle('Распределение данных Тип здания и Тип жилья')
plt.show()


# In[108]:


# Рис. 2.2. Диаграмма Бокс-Виксера для качественных переменных до удаления выбросов (стр. 6)
fig = plt.figure(figsize=(15, 15))
sns.boxplot(x='building_type', y='price_per_metr', data=DB, palette='Blues_r')
plt.ylabel('цена за кв.м')


# In[109]:


# Рис. 2.1. Диаграмма Бокс-Виксера для качественных переменных до удаления выбросов (стр. 5)
# используя библиотеку Seaborn строим boxplot 
fig = plt.figure(figsize=(15, 15))
sns.boxplot(x='object_type', y='price_per_metr', data=DB, palette='Blues_r', width=0.5)
plt.ylabel('Цена за кв.м')
plt.xlabel('Тип здания', fontsize=16)


# In[32]:


# Рисунок 5. Диаграмма Бокса-Уискера. Зависимость цены за м² от типа жилья (стр. 10)
# используя библиотеку Seaborn строим boxplot 
fig = plt.figure(figsize=(15, 15))
sns.boxplot(x='Тип жилья', y='price_per_metr', data=dvlp, palette='Blues_r')
plt.ylabel('цена за кв.м')


# In[33]:


# Рисунок 4. Диаграмма Бокса-Уискера. Зависимость цены за м² от типа здания (стр. 10)
# используя библиотеку Seaborn строим boxplot 
fig = plt.figure(figsize=(15, 15))
sns.boxplot(x='Тип здания', y='price_per_metr', data=dvlp, palette='Blues_r', width=0.5)
plt.ylabel('Цена за кв.м')
plt.xlabel('Тип здания', fontsize=16)


# In[34]:


# Рисунок 6. Зависимость целевой переменной от объясняющих (стр. 11)
# используя библиотеку Seaborn строим диаграмму рассеяния ("точечную диаграмму")
fig, axes = plt.subplots(1, figsize = (6.5, 6.5)) # указали кол-во фигур на листе, размер фигуры
sns.set()
fig.suptitle('Тюмень') # указали название 

sns.scatterplot(x='levels', y='price_per_metr', data=tym_ad, alpha = 0.02, color='darkblue') # прописали параметры: переменную х, у, данные, блюр, цвет
plt.ylabel('цена за кв.м') # добавили имя для оси ОУ
plt.xlabel('этажность дома') # добавили имя для оси ОХ
pl


# In[35]:


# Рисунок 6. Зависимость целевой переменной от объясняющих (стр. 11)
fig, axes = plt.subplots(1, figsize = (6.5, 6.5))
sns.set()
fig.suptitle('Тюмень')

sns.scatterplot(x='level', y='price_per_metr', data=tym_ad, alpha = 0.02, color='darkblue')
plt.ylabel('цена за кв.м')
plt.xlabel('этаж')


# In[36]:


# Рисунок 6. Зависимость целевой переменной от объясняющих (стр. 11)
fig, axes = plt.subplots(1, figsize = (6.5, 6.5))
sns.set()
fig.suptitle('Тюмень')

sns.scatterplot(x='rooms', y='price_per_metr', data=tym_ad, alpha = 0.02, color='darkblue')
plt.ylabel('цена за кв.м')
plt.xlabel('количество комнат')


# In[37]:


# Рисунок 6. Зависимость целевой переменной от объясняющих (стр. 11)
fig, axes = plt.subplots(1, figsize = (6.5, 6.5))
sns.set()
fig.suptitle('Тюмень')

sns.scatterplot(x='area', y='price_per_metr', data=tym_ad, alpha = 0.02, color='darkblue')
plt.ylabel('цена за кв.м')
plt.xlabel('площадь в м^2')


# In[38]:


# Рисунок 6. Зависимость целевой переменной от объясняющих (стр. 11)
fig, axes = plt.subplots(1, figsize = (6.5, 6.5))
sns.set()
fig.suptitle('Тюмень')

sns.scatterplot(x='kitchen_area', y='price_per_metr', data=tym_ad, alpha = 0.02, color='darkblue')
plt.ylabel('цена за кв.м')
plt.xlabel('площадь кухни в м^2')


# In[39]:


# Рисунок 6. Зависимость целевой переменной от объясняющих (стр. 11)
fig, axes = plt.subplots(1, figsize = (6.5, 6.5))
sns.set()
fig.suptitle('Тюмень')

sns.scatterplot(x='Тип жилья', y='price_per_metr', data=dvlp, alpha = 0.02, color='darkblue') # данные СА не нужно переименовывать, в рамках задачи они равны tym_ad
plt.ylabel('цена за кв.м')
plt.xlabel('Тип жилья')


# In[110]:


# Рисунок 6. Зависимость целевой переменной от объясняющих (стр. 11)
fig, axes = plt.subplots(1, figsize = (6.5, 6.5))
sns.set()
fig.suptitle('Тюмень')

sns.scatterplot(x='distance', y='price_per_metr', data=tym_ad, alpha = 0.02, color='darkblue')
plt.ylabel('цена за кв.м')
plt.xlabel('отдаленность от центра')


# In[111]:


# Рисунок.1.2. После удаления выбросов и фильтрации (стр. 7)
fig, axes = plt.subplots(2, figsize = (20, 20)) # прописали условия для листа

plt.subplot(3, 3, 1) # расположение гарфика
plt.xlim([0, 250]) # установили лимит для понятной визуализации: так данные будут более подробны
sns.histplot(
    tym_ad["area"], kde=True,
    stat="density", color='darkblue') # прописали условия для графика согласно документации с оф. сайта Seaborn 
plt.ylabel('') # нам не нужна подпись по ОУ
plt.xlabel('площадь') # подпись по ОХ

# аналогично 

plt.subplot(3, 3, 2)
plt.subplots_adjust(hspace = 0.5)
sns.histplot(
    tym_ad['level'], kde=True,
    stat="density", color='darkblue')
plt.ylabel('')
plt.xlabel('этаж')


plt.subplot(3, 3, 4)
sns.histplot(
    tym_ad['levels'], kde=True,
    stat="density", kde_kws=dict(cut=3), color='darkblue')
plt.ylabel('')
plt.xlabel('этажность')
plt.xlim([0, 30])

plt.subplot(3, 3, 3)
sns.histplot(
    tym_ad['kitchen_area'], kde=True,
    stat="density", kde_kws=dict(cut=3), color='darkblue')
plt.xlim([0, 30])
plt.ylabel('')
plt.xlabel('площадь кухни в м^2')

plt.subplot(3, 3, 5)
sns.histplot(
    tym_ad['distance'], kde=True,
    stat="density", kde_kws=dict(cut=3), color='darkblue')
plt.xlim([0, 20])
plt.ylabel('')
plt.xlabel('отдаленность от центра')

plt.subplot(3, 3, 6)
sns.histplot(
    tym_ad['price_per_metr'], kde=True,
    stat="density", kde_kws=dict(cut=3), color='darkblue')
plt.ylabel('')
plt.xlabel('цена за м^2')


# In[112]:


# Рисунок 7. Сила связи между независимыми переменными по Спирмену и Пирсону (стр. 12)
fig, axes = plt.subplots(1, figsize = (25, 25)) # прописали условия для листа
plt.subplot(2, 2, 2) # расположение
plt.title('Пирсон')
sns.heatmap(C_P.corr(), annot = True, cmap='Blues') # прописали параметры для графика

plt.subplot(2, 2, 1) 
plt.title('Спирмен') 
sns.heatmap(C_S.corr(), annot = True, cmap='Blues')


# In[43]:


fig, axes = plt.subplots(2, figsize = (20, 20)) # прописали условия для листа

plt.subplot(3, 3, 1) # расположение гарфика
plt.xlim([0, 250]) # установили лимит для понятной визуализации: так данные будут более подробны
sns.histplot(
    data_describe["area"], kde=True,
    stat="density", color='darkblue') # прописали условия для графика согласно документации с оф. сайта Seaborn 
plt.ylabel('') # нам не нужна подпись по ОУ
plt.xlabel('площадь') # подпись по ОХ

# аналогично 

plt.subplot(3, 3, 2)
plt.subplots_adjust(hspace = 0.5)
sns.histplot(
    data_describe['level'], kde=True,
    stat="density", color='darkblue')
plt.ylabel('')
plt.xlabel('этаж')


plt.subplot(3, 3, 4)
sns.histplot(
    data_describe['levels'], kde=True,
    stat="density", kde_kws=dict(cut=3), color='darkblue')
plt.ylabel('')
plt.xlabel('этажность')
plt.xlim([0, 30])

plt.subplot(3, 3, 3)
sns.histplot(
    data_describe['kitchen_area'], kde=True,
    stat="density", kde_kws=dict(cut=3), color='darkblue')
plt.xlim([0, 30])
plt.xlabel('площадь кухни в м^2')


plt.subplot(3, 3, 5)
sns.histplot(
    data_describe['price_per_metr'], kde=True,
    stat="density", kde_kws=dict(cut=3), color='darkblue')
plt.xlabel('цена за м^2')


plt.savefig('/Users/polinapashchenko/Desktop/work/graphics/histplot_before.pdf', format='pdf') # сохранили


# In[124]:


# Меняем тип данных
tym_mod = tym_ad.astype({'price_per_metr':np.float64, 'building_type':'category', 'level':np.float64, 'levels':np.float64,
                      'object_type':'category', 'rooms':np.float64})
tym_mod.info()

# Таблица 4. Оценка базовой модели (стр. 13)
# Таблица 5. Оценка базовой модели после удаления некоторых количественных переменных (стр. 14)
# Таблица 6. Оценка базовой модели после удаления качественной переменных (стр. 15)
# Таблица 7.  Оценка модели для проверки гипотезы 1 (стр. 16)
# Таблица 8. Оценка модели для проверки гипотезы 2 (стр. 16)
# Таблица 9. Оценка модели, учитывающая две сложные гипотезы (стр. 17)

# Разбиение данных на тренировочное и тестовое множество
# random_state - для повторного отбора тех же элементов
tym_mod_train = tym_mod.sample(frac=0.8, random_state=42) 
tym_mod_test = tym_mod.loc[~tym_mod.index.isin(tym_mod_train.index)] 

# Накапливаем данные о качестве постреонных моделей
# Используем  adjR^2 и AIC
mq = pd.DataFrame([], columns=['adjR^2', 'AIC']) # Данные о качестве


# In[132]:


get_ipython().system('pip install statsmodels')


# In[134]:


import statsmodels.discrete.discrete_model 
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS, WLS


# In[136]:


# Формируем целевую переменную - price_per_metr
Y = tym_mod_train['price_per_metr']
# Формируем фиктивные переменные для всех качественных переменных
DUM = pd.get_dummies(tym_mod_train[['object_type', 'building_type']])
# Выбираем переменные для уровней, которые войдут в модель
DUM = DUM[['building_type_1', 'building_type_2', 'building_type_3', 'building_type_4', 'building_type_5', 'object_type_0']]
# Формируем pandas.DataFramee содержащий матрицу X объясняющих переменных 
# Добавляем слева фиктивные переменные
X_1 = DUM.concat([DUM, tym_mod_train[['level', 'levels', 'area', 'rooms', 'kitchen_area', 'distance']]], axis=1)
# Добавляем переменную равную единице для учета константы
X_1 = sm.add_constant(X_1)
X_1 = X_1.astype({'const':'uint8'}) # Сокращаем место для хранения константы
# Формируем объект, содержащй все исходные данные и методы для оценивания
linreg01 = sm.OLS(YX_1)
# Оцениваем модель
fitmod01 = linreg01.fit()
# Сохраняем результаты оценки в файл
with open('develop_STAT_1.txt', 'a') as fln:
    print('\n ****** Оценка базовой модели ******',
          file=fln)
    print(fitmod01.summary(), file=fln)
# Модель имеет мультиколлинеарность и обладает гетероскедастичностью, незначимых переменных нет


# In[138]:


# Проверяем степень мультиколлинеарности базовой модели
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame() # Для хранения 
X_q = X_1.select_dtypes(include=['float64']) # Только количественные регрессоры
vif["vars"] = X_q.columns
vif["VIF"] = [variance_inflation_factor(X_q.values, i) 
              for i in range(X_q.shape[1])]
# Сохраняем полученные результаты
with pd.ExcelWriter('develop_STAT_1.xlsx', engine="openpyxl") as wrt:
    vif.to_excel(wrt, sheet_name='vif')


# In[139]:


# Проверяем гетероскедастичность базовой модели
# помощью коритерия White(а) и F критерия
from statsmodels.stats.diagnostic import het_white
e = fitmod01.resid
WHT = pd.DataFrame(het_white(e, X_1), index= ['LM', 'LM_P', 'F', 'F_P'])
# Сохраняем полученные результаты
with pd.ExcelWriter('/Users/polinapashchenko/Desktop/work/output/develop_STAT_1.xlsx', engine="openpyxl", 
                    if_sheet_exists='overlay', mode='a') as wrt:
    WHT.to_excel(wrt, sheet_name='het')


# In[140]:


# Сохраняем данные о качестве модели
q = pd.DataFrame([fitmod01.rsquared_adj, fitmod01.aic], 
                 index=['adjR^2', 'AIC'], columns=['base_00']).T
mq = pd.concat([mq, q])  


# In[141]:


# Убираем переменную area, как она имеет вздутую дисперсию
X_2 = X_1.drop(['area'], axis=1)
# Формируем объект, содержащий все исходные данные и методы для оценивания
linreg02 = sm.OLS(Y,X_2)
# Оцениваем модель
fitmod02 = linreg02.fit()
# Сохраняем результаты оценки в файл
with open('/Users/polinapashchenko/Desktop/work/output/develop_STAT_2.txt', 'a') as fln:
    print('\n ****** Оценка базовой модели ******',
          file=fln)
    print(fitmod02.summary(), file=fln)
# Модель имеет мультиколлинеарность и обладает гетероскедастичностью, незначимых переменных нет. 
# Качество модели осталось на таком же уровне


# In[142]:


# Проверяем степень мультиколлинеарности базовой модели
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame() # Для хранения 
X_q = X_2.select_dtypes(include='float64')# Только количественные регрессоры
vif["vars"] = X_q.columns
vif["VIF"] = [variance_inflation_factor(X_q.values, i) 
              for i in range(X_q.shape[1])]
# Сохраняем полученные результаты
with pd.ExcelWriter('/Users/polinapashchenko/Desktop/work/output/develop_STAT_2.xlsx', engine="openpyxl") as wrt:
    vif.to_excel(wrt, sheet_name='vif')


# In[143]:


# Проверяем гетероскедастичность базовой модели
# помощью коритерия White(а) и F критерия
from statsmodels.stats.diagnostic import het_white
e = fitmod02.resid
WHT = pd.DataFrame(het_white(e, X_2), index= ['LM', 'LM_P', 'F', 'F_P'])
# Сохраняем полученные результаты
with pd.ExcelWriter('/Users/polinapashchenko/Desktop/work/output/develop_STAT_2.xlsx', engine="openpyxl", 
                    if_sheet_exists='overlay', mode='a') as wrt:
    WHT.to_excel(wrt, sheet_name='het')


# In[144]:


# Сохраняем данные о качестве модели
q = pd.DataFrame([fitmod02.rsquared_adj, fitmod02.aic], 
                 index=['adjR^2', 'AIC'], columns=['base_01']).T
mq = pd.concat([mq, q])  


# In[145]:


# Убираем переменную level, так как level и levels обладают вздутой дисперсией, 
# и имеют сильную положительную корреляционную связь (0.71)
# Так как levels сильнее коррелирует с целевой переменной, сначала попробуем убрать level
X_3 = X_2.drop(['level'], axis=1)
# Формируем объект, содержащий все исходные данные и методы для оценивания
linreg03 = sm.OLS(Y,X_3)
# Оцениваем модель
fitmod03 = linreg03.fit()
# Сохраняем результаты оценки в файл
with open('/Users/polinapashchenko/Desktop/work/output/develop_STAT_3.txt', 'a') as fln:
    print('\n ****** Оценка базовой модели ******',
          file=fln)
    print(fitmod03.summary(), file=fln)
# Модель имеет мультиколлинеарность и обладает гетероскедастичностью, незначимых переменных нет
# Качество модели осталось на таком же уровне


# In[146]:


# Проверяем степень мультиколлинеарности базовой модели
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame() # Для хранения 
X_q = X_3.select_dtypes(include='float64')# Только количественные регрессоры
vif["vars"] = X_q.columns
vif["VIF"] = [variance_inflation_factor(X_q.values, i) 
              for i in range(X_q.shape[1])]
# Сохраняем полученные результаты
with pd.ExcelWriter('/Users/polinapashchenko/Desktop/work/output/develop_STAT_3.xlsx', engine="openpyxl") as wrt:
    vif.to_excel(wrt, sheet_name='vif')


# In[147]:


# Проверяем гетероскедастичность базовой модели
# помощью коритерия White(а) и F критерия
from statsmodels.stats.diagnostic import het_white
e = fitmod03.resid
WHT = pd.DataFrame(het_white(e, X_3), index= ['LM', 'LM_P', 'F', 'F_P'])
# Сохраняем полученные результаты
with pd.ExcelWriter('/Users/polinapashchenko/Desktop/work/output/develop_STAT_3.xlsx', engine="openpyxl", 
                    if_sheet_exists='overlay', mode='a') as wrt:
    WHT.to_excel(wrt, sheet_name='het')


# In[148]:


# Сохраняем данные о качестве модели
q = pd.DataFrame([fitmod03.rsquared_adj, fitmod03.aic], 
               index=['adjR^2', 'AIC'], columns=['base_02']).T
mq = pd.concat([mq, q]) 


# In[92]:


# В модели все еще присутствовала мультиколлинеарность и гетероскедастичность, 
# а kitchen_area, levels и rooms, хоть и имеют вздутую дисперсию, необходимы нам для проверки гипотез
# поэтому попробуем убрать последнюю колиечтсвенную переменную distance (хотя она значима и не имеет взудую дисперсию) 
# и посмотреть на качество и свойства модели
X_p = X_3.drop(['distance'], axis=1)
# Формируем объект, содержащий все исходные данные и методы для оценивания
linreg0p = sm.OLS(Y,X_p)
# Оцениваем модель
fitmod0p = linreg0p.fit()
# Сохраняем результаты оценки в файл
with open('/Users/polinapashchenko/Desktop/work/output/develop_STAT_p.txt', 'a') as fln:
    print('\n ****** Оценка базовой модели ******',
          file=fln)
    print(fitmod0p.summary(), file=fln)
# Качество модели ухудшилось, она все еще мультиколлинеарна и имеет гетероскедастичность, вернемся к предыдущей модели


# In[93]:


# Проверяем степень мультиколлинеарности базовой модели
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame() # Для хранения 
X_q = X_p.select_dtypes(include='float64')# Только количественные регрессоры
vif["vars"] = X_q.columns
vif["VIF"] = [variance_inflation_factor(X_q.values, i) 
              for i in range(X_q.shape[1])]
# Сохраняем полученные результаты
with pd.ExcelWriter('/Users/polinapashchenko/Desktop/work/output/develop_STAT_p.xlsx', engine="openpyxl") as wrt:
    vif.to_excel(wrt, sheet_name='vif')


# In[94]:


# Проверяем гетероскедастичность базовой модели
# помощью коритерия White(а) и F критерия
from statsmodels.stats.diagnostic import het_white
e = fitmod0p.resid
WHT = pd.DataFrame(het_white(e, X_p), index= ['LM', 'LM_P', 'F', 'F_P'])
# Сохраняем полученные результаты
with pd.ExcelWriter('/Users/polinapashchenko/Desktop/work/output/develop_STAT_p.xlsx', engine="openpyxl", 
                    if_sheet_exists='overlay', mode='a') as wrt:
    WHT.to_excel(wrt, sheet_name='het')


# In[95]:


# Сохраняем данные о качестве модели
q = pd.DataFrame([fitmod0p.rsquared_adj, fitmod0p.aic], 
               index=['adjR^2', 'AIC'], columns=['base_03']).T
mq = pd.concat([mq, q]) 


# In[96]:


# Ввернемся к 3 модели, все остальные переменные нам нужны для проверки гипотез, кроме качественных переменных building_type,
# попробуем их удалить и посмотреть на качество и свойства модели
X_k = X_3.drop(['building_type_1', 'building_type_2', 'building_type_3', 'building_type_4', 'building_type_5'], axis=1)
# Формируем объект, содержащий все исходные данные и методы для оценивания
linreg0k = sm.OLS(Y,X_k)
# Оцениваем модель
fitmod0k = linreg0k.fit()
# Сохраняем результаты оценки в файл
with open('/Users/polinapashchenko/Desktop/work/output/develop_STAT_k.txt', 'a') as fln:
    print('\n ****** Оценка базовой модели ******',
          file=fln)
    print(fitmod0k.summary(), file=fln)
# Качество модели ухудшилось (R^2 уменьшился до 0.94, AIC незначительно увеличился) и имеет гетероскедастичность, 
# вернемся к предыдущей модели (3), так как она обладала лучшими метриками


# In[97]:


# Сохраняем данные о качестве модели
q = pd.DataFrame([fitmod0k.rsquared_adj, fitmod0k.aic], 
               index=['adjR^2', 'AIC'], columns=['base_04']).T
mq = pd.concat([mq, q]) 


# In[98]:


# Вернемся к 3 модели, и оставим как лучшую
X_final = X_3
# Формируем объект, содержащий все исходные данные и методы для оценивания
linreg04 = sm.OLS(Y,X_final)
# Оцениваем модель
fitmod04 = linreg04.fit()
# Сохраняем результаты оценки в файл
with open('/Users/polinapashchenko/Desktop/work/output/develop_STAT_final.txt', 'a') as fln:
    print('\n ****** Оценка базовой модели ******',
          file=fln)
    print(fitmod04.summary(), file=fln)
# Качество модели немного улучшилось, однако осталась мультиколлинеарность, гетеросекедастичность также присутствует.
# На данном этапе оставим модель в таком виде. Переменные, которые имеют вздутую дисперсию (не критичную) и незначима - 
# нужны нам для проверки гипотез, поэтому мы их оставим
# Все переменные значимы


# In[99]:


# Проверяем степень мультиколлинеарности базовой модели
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame() # Для хранения 
X_q = X_final.select_dtypes(include='float64')# Только количественные регрессоры
vif["vars"] = X_q.columns
vif["VIF"] = [variance_inflation_factor(X_q.values, i) 
              for i in range(X_q.shape[1])]
# Сохраняем полученные результаты
with pd.ExcelWriter('/Users/polinapashchenko/Desktop/work/output/develop_STAT_final.xlsx', engine="openpyxl") as wrt:
    vif.to_excel(wrt, sheet_name='vif')


# In[100]:


# Проверяем гетероскедастичность базовой модели
# помощью коритерия White(а) и F критерия
from statsmodels.stats.diagnostic import het_white
e = fitmod04.resid
WHT = pd.DataFrame(het_white(e, X_final), index= ['LM', 'LM_P', 'F', 'F_P'])
# Сохраняем полученные результаты
with pd.ExcelWriter('/Users/polinapashchenko/Desktop/work/output/develop_STAT_final.xlsx', engine="openpyxl", 
                    if_sheet_exists='overlay', mode='a') as wrt:
    WHT.to_excel(wrt, sheet_name='het')


# In[101]:


# Сохраняем данные о качестве модели
q = pd.DataFrame([fitmod04.rsquared_adj, fitmod04.aic], 
               index=['adjR^2', 'AIC'], columns=['base_05']).T
mq = pd.concat([mq, q]) 


# In[102]:


# проверяем гипотезу о том, что сила влияния площади кухни на цену за кв.м зависит от типа жилья
# в новостройках площадь кухни имеет большее влияние на цену, чем во вторичке
# Вводим переменную взаимодействия
X_h_1 = X_final.copy()
X_h_1['ok'] = X_h_1['kitchen_area']*X_h_1['object_type_0']
linreg001 = sm.OLS(Y,X_h_1)
fitmod001 = linreg001.fit()
# Сохраняем результаты оценки в файл
with open('/Users/polinapashchenko/Desktop/work/output/develop_STAT_h.txt', 'a') as fln:
    print('\n ****** Оценка базовой модели ******',
          file=fln)
    print(fitmod001.summary(), file=fln)
    
# Сохраняем данные о качестве модели
q = pd.DataFrame([fitmod001.rsquared_adj, fitmod001.aic], 
                 index=['adjR^2', 'AIC'], columns=['hyp_01']).T
mq = pd.concat([mq, q])    


# In[103]:


# проверяем гипотезу о том, что сила влияния количества комнат на цену за кв.м зависит от количества комнат
# До определенного количества комнат цена за кв. м падает быстрее
# Вводим переменную взаимодействия
thr = 3 # Порог количества комнат
X_h_2 = X_final.copy()
# Формируем dummy из качественных переменных
rooms_thr = X_h_2['rooms'] >= thr
X_h_2['r'] = X_h_2['rooms']*rooms_thr # Взаимодействие
linreg002 = sm.OLS(Y,X_h_2)
fitmod002 = linreg002.fit()
# Сохраняем результаты оценки в файл
with open('develop_STAT_h.txt', 'a') as fln:
    print('\n ****** Оценка базовой модели ******',
          file=fln)
    print(fitmod002.summary(), file=fln)
    
# Сохраняем данные о качестве модели
q = pd.DataFrame([fitmod002.rsquared_adj, fitmod002.aic], 
                 index=['adjR^2', 'AIC'], columns=['hyp_02']).T
mq = pd.concat([mq, q]) 


# In[104]:


# Построим модель, в которой присутствуют обе гипотезы
# Вводим переменную взаимодействия
X_h_12 = X_final.copy()
X_h_12['ok'] = X_h_12['object_type_0']*X_h_12['kitchen_area']
rooms_thr = X_h_12['rooms'] >= thr
X_h_12['r'] = X_h_12['rooms']*rooms_thr # Взаимодействие
linreg003 = sm.OLS(Y,X_h_12)
fitmod003 = linreg003.fit()
# Сохраняем результаты оценки в файл
with open('develop_STAT_h.txt', 'a') as fln:
    print('\n ****** Оценка базовой модели ******',
          file=fln)
    print(fitmod003.summary(), file=fln)
    
# Сохраняем данные о качестве модели
q = pd.DataFrame([fitmod003.rsquared_adj, fitmod003.aic], 
                 index=['adjR^2', 'AIC'], columns=['hyp_03']).T
mq = pd.concat([mq, q])   


# In[119]:


# Предсказательная сила
Y_test = tym_mod_test['price_per_metr']
X_test = X_h_12
# Добавляем переменную равную единице для учета константы
X_test = sm.add_constant(X_test)
X_test = X_test.astype({'const':'uint8'})
# Генерация предсказаний на тестовом множестве 
pred_ols = fitmod003.get_prediction(X_test)
# Генерация доверительных интервалов с доверительной вероятностью alpha
frm = pred_ols.summary_frame(alpha=0.05)
iv_l = frm["obs_ci_lower"] # Нижняя граница доверительных интервалов
iv_u = frm["obs_ci_upper"] # Верхняя граница доверительных интервалов
fv = frm['mean'] # Предсказанное значение целевой переменной
# Построение графиков
name = 'ok' # Имя переменной относительно которой строим прогноз
Z = X_test.loc[:, name]
dfn = pd.DataFrame([Z, Y_test, fv, iv_u, iv_l]).T
dfn = dfn.sort_values(by=name)
fig, ax = plt.subplots(figsize=(8, 6))
for z in dfn.columns[1:]:
    dfn.plot(x=dfn.columns[0], y=z, ax=ax)
ax.legend(loc="best")
plt.show()

# Подсчет среднеквадратической ошибки
dif = np.sqrt((dfn.iloc[:,1] - dfn.iloc[:,2]).pow(2).sum()/dfn.shape[0])
print(dif)
# Доля выходов за границы доверительых интервалов
# Сортируем, чтобы индексы во всех рядах совпадали
mn = dfn.iloc[:,1].sort_index() 
out = ((mn > iv_u) + (mn < iv_l)).sum()/dfn.shape[0]


# In[ ]:


# Предсказательная сила
Y_test = tym_mod_test['price_per_metr']
X_test = X_h_12
# Добавляем переменную равную единице для учета константы
X_test = sm.add_constant(X_test)
X_test = X_test.astype({'const':'uint8'})
# Генерация предсказаний на тестовом множестве 
pred_ols = fitmod003.get_prediction(X_test)
# Генерация доверительных интервалов с доверительной вероятностью alpha
frm = pred_ols.summary_frame(alpha=0.05)
iv_l = frm["obs_ci_lower"] # Нижняя граница доверительных интервалов
iv_u = frm["obs_ci_upper"] # Верхняя граница доверительных интервалов
fv = frm['mean'] # Предсказанное значение целевой переменной
# Построение графиков
name = 'r' # Имя переменной относительно которой строим прогноз
Z = X_test.loc[:, name]
dfn = pd.DataFrame([Z, Y_test, fv, iv_u, iv_l]).T
dfn = dfn.sort_values(by=name)
fig, ax = plt.subplots(figsize=(8, 6))
for z in dfn.columns[1:]:
    dfn.plot(x=dfn.columns[0], y=z, ax=ax)
ax.legend(loc="best")
plt.show()

# Подсчет среднеквадратической ошибки
dif = np.sqrt((dfn.iloc[:,1] - dfn.iloc[:,2]).pow(2).sum()/dfn.shape[0])
print(dif)
# Доля выходов за границы доверительых интервалов
# Сортируем, чтобы индексы во всех рядах совпадали
mn = dfn.iloc[:,1].sort_index() 
out = ((mn > iv_u) + (mn < iv_l)).sum()/dfn.shape[0]


# In[ ]:


# Предсказательная сила
Y_test = tym_mod_test['price_per_metr']
X_test = X_h_12
# Добавляем переменную равную единице для учета константы
X_test = sm.add_constant(X_test)
X_test = X_test.astype({'const':'uint8'})
# Генерация предсказаний на тестовом множестве 
pred_ols = fitmod003.get_prediction(X_test)
# Генерация доверительных интервалов с доверительной вероятностью alpha
frm = pred_ols.summary_frame(alpha=0.05)
iv_l = frm["obs_ci_lower"] # Нижняя граница доверительных интервалов
iv_u = frm["obs_ci_upper"] # Верхняя граница доверительных интервалов
fv = frm['mean'] # Предсказанное значение целевой переменной
# Построение графиков
name = 'levels' # Имя переменной относительно которой строим прогноз
Z = X_test.loc[:, name]
dfn = pd.DataFrame([Z, Y_test, fv, iv_u, iv_l]).T
dfn = dfn.sort_values(by=name)
fig, ax = plt.subplots(figsize=(8, 6))
for z in dfn.columns[1:]:
    dfn.plot(x=dfn.columns[0], y=z, ax=ax)
ax.legend(loc="best")
plt.show()

# Подсчет среднеквадратической ошибки
dif = np.sqrt((dfn.iloc[:,1] - dfn.iloc[:,2]).pow(2).sum()/dfn.shape[0])
print(dif)
# Доля выходов за границы доверительых интервалов
# Сортируем, чтобы индексы во всех рядах совпадали
mn = dfn.iloc[:,1].sort_index() 
out = ((mn > iv_u) + (mn < iv_l)).sum()/dfn.shape[0]

