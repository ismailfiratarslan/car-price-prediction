import streamlit as st

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")
from xgboost import XGBRegressor


st.title("Araba Fiyat Tahmini")


df = pd.read_csv("cars-last.csv")


df = df.drop(["İlan No", "İlan Tarihi", "Motor Hacmi", "Ort. Yakıt Tüketimi", "Yakıt Deposu"], axis=1)
# df = df.drop(["İlan No", "İlan Tarihi", "Ort. Yakıt Tüketimi", "Yakıt Deposu"], axis=1)
df = df.drop_duplicates()

df = df.dropna()


df["Kilometre"] = df["Kilometre"].apply(lambda x: x.replace(".", ""))
df["Kilometre"] = df["Kilometre"].apply(lambda x: x.strip("km"))
df["Kilometre"] = df["Kilometre"].astype(int)

df = df[~df['Motor Gücü'].str.contains("1598 cc|1597 cc|ve|-")]
df["Motor Gücü"] = df["Motor Gücü"].apply(lambda x: x.strip("hp"))
df["Motor Gücü"] = df["Motor Gücü"].astype(int)


df_copy = df.copy()

last_seri = df_copy.sort_values(['Seri'])
last_model = df_copy.sort_values(['Model'])
last_degisenler = df_copy.sort_values(['Boya-değişen'])

last_degisenler_boyali = last_degisenler[last_degisenler["Boya-değişen"].str.contains(",|Tamamı orjinal|Belirtilmemiş|değişen|66 lt|68 lt") == False]
last_degisenler_degisen = last_degisenler[last_degisenler["Boya-değişen"].str.contains(",|Tamamı orjinal|Belirtilmemiş|boyalı|66 lt|68 lt") == False]
last_degisenler_boyali_degisen = last_degisenler[last_degisenler["Boya-değişen"].str.contains(",")]

lines = last_degisenler_boyali_degisen["Boya-değişen"].tolist()

newlist = []
for word in lines:
    word = word.split(",")
    newlist.extend(word)

newlist = list(dict.fromkeys(newlist))

ekle_boyali = ['boyalı']
ekle_degisen = ['değişen']
ekle_boyali_list = []
ekle_degisen_list = []

for find in newlist:
    if any(item in find for item in ekle_boyali):
        ekle_boyali_list.append(find)
    if any(item in find for item in ekle_degisen):
        ekle_degisen_list.append(find)

ekle_boyali_list = [x.strip() for x in ekle_boyali_list]

############################################################################
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

############################################################################

le = LabelEncoder()

df["Marka"] = le.fit_transform(df["Marka"])
dict_Marka = dict(zip(le.classes_, le.transform(le.classes_)))

df["Seri"] = le.fit_transform(df["Seri"])
dict_Seri = dict(zip(le.classes_, le.transform(le.classes_)))

df["Model"] = le.fit_transform(df["Model"])
dict_Model = dict(zip(le.classes_, le.transform(le.classes_)))

df["Vites Tipi"] = le.fit_transform(df["Vites Tipi"])
dict_VitesTipi = dict(zip(le.classes_, le.transform(le.classes_)))

df["Yakıt Tipi"] = le.fit_transform(df["Yakıt Tipi"])
dict_YakıtTipi = dict(zip(le.classes_, le.transform(le.classes_)))

df["Kasa Tipi"] = le.fit_transform(df["Kasa Tipi"])
dict_KasaTipi = dict(zip(le.classes_, le.transform(le.classes_)))

df["Çekiş"] = le.fit_transform(df["Çekiş"])
dict_Cekis = dict(zip(le.classes_, le.transform(le.classes_)))

df["Boya-değişen"] = le.fit_transform(df["Boya-değişen"])
dict_BoyaDegisen = dict(zip(le.classes_, le.transform(le.classes_)))

df["Takasa Uygun"] = le.fit_transform(df["Takasa Uygun"])
dict_TakasaUygun = dict(zip(le.classes_, le.transform(le.classes_)))

df["Kimden"] = le.fit_transform(df["Kimden"])
dict_Kimden= dict(zip(le.classes_, le.transform(le.classes_)))



x = df.iloc[:, :-1].values # bağımsız değişkenler
y = df.iloc[:, -1:].values # bağımlı değişkenler

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=100)


xgb = XGBRegressor()
xgb.fit(x_train, y_train)


brand = st.selectbox("Marka:", dict_Marka.keys())

seri_list = dict.fromkeys(df_copy.loc[df_copy['Marka'] == brand]['Seri'].tolist())
seri_list_set = set(seri_list)
dict_Seri_set = set(dict_Seri)
seri_list_ = []

for name in seri_list_set.intersection(dict_Seri_set):
    seri_list_.append(name)

series = st.selectbox("Seri:", sorted(seri_list_))

model_list = dict.fromkeys(df_copy.loc[df_copy['Seri'] == series]['Model'].tolist())
model_list_set = set(model_list)
dict_Model_set = set(dict_Model)
model_list_ = []

for name in model_list_set.intersection(dict_Model_set):
    model_list_.append(name)

model = st.selectbox("Model:", sorted(model_list_))

productionYear = st.number_input("Üretim Yılı:", format="%d", value=0)
mileage = st.number_input("Kilometre:", format="%d", value=0)

sortedgearboxList = ['Düz', 'Otomatik', 'Yarı Otomatik']

gearbox = st.selectbox("Vites tipi:", sortedgearboxList)

fuelTypeList = dict.fromkeys(df_copy.loc[df_copy['Model'] == model]['Yakıt Tipi'].tolist())
fuelTypeList_set = set(fuelTypeList)
dict_YakıtTipi_set = set(dict_YakıtTipi)
fuelTypeList_ = []

for name in fuelTypeList_set.intersection(dict_YakıtTipi_set):
    fuelTypeList_.append(name)

if fuelTypeList_[0] == "LPG & Benzin":
    fuelTypeList_.append("Benzin")

fuelType = st.selectbox("Yakıt türü:", fuelTypeList_)

bodyTypeListFrame = df_copy[df_copy["Kasa Tipi"].str.contains("Dizel") == False]
bodyTypeList = dict.fromkeys(bodyTypeListFrame.loc[bodyTypeListFrame['Seri'] == series]['Kasa Tipi'].tolist())
sortedbodyTypeList = sorted(bodyTypeList.keys(), key=lambda x:x.lower())

bodyType = st.selectbox("Kasa tipi:", sortedbodyTypeList)
enginePower = st.number_input("Motor gücü:", format="%d", value=0)

drivetrainFrame = df_copy[df_copy["Çekiş"].str.contains("120 hp|170 hp") == False]
drivetrainList = dict.fromkeys(drivetrainFrame.loc[drivetrainFrame['Model'] == model]['Çekiş'].tolist())
sorteddrivetrainList = sorted(drivetrainList.keys(), key=lambda x:x.lower())

drivetrain = st.selectbox("Çekiş:", sorteddrivetrainList)


replacedParts = st.selectbox("Boya-değişen:", ["Tamamı orjinal", "Sadece Boyalı", "Sadece Değişen", "Hem Boyalı Hem Değişenli", "Belirtilmemiş"])

newReplacedParts = st.empty()
newReplacedParts2 = st.empty()

if replacedParts == "Tamamı orjinal":
    replacedParts = replacedParts

elif replacedParts == "Sadece Boyalı":
    replacedParts = newReplacedParts.selectbox("Boyalı Parça Sayısı:", dict.fromkeys(last_degisenler_boyali["Boya-değişen"].tolist()))
elif replacedParts == "Sadece Değişen":
    replacedParts = newReplacedParts.selectbox("Değişen Parça Sayısı:", dict.fromkeys(last_degisenler_degisen["Boya-değişen"].tolist()))
elif replacedParts == "Hem Boyalı Hem Değişenli":
    boyaliSayisi = newReplacedParts.selectbox("Boyalı Parça Sayısı:", ekle_boyali_list)
    degisenSayisi = newReplacedParts2.selectbox("Değişen Parça Sayısı:", ekle_degisen_list)
    replacedParts = degisenSayisi + ", " + boyaliSayisi
else:
    replacedParts = "Belirtilmemiş"

exchange = st.radio("Takasa uygunluk:", ["Takasa Uygun", "Takasa Uygun Değil"])
fromWhom = st.radio("Kimden:", ["Sahibinden", "Galeriden"])
predict = st.button("Predict!")

if predict:

    print("Marka:", brand)
    print("Seri:", series)
    print("Model:", model)
    print("Üretim Yılı:", productionYear)
    print("Kilometre:", mileage)
    print("Vites tipi:", gearbox)
    print("Yakıt türü:", fuelType)
    print("Kasa tipi:", bodyType)
    print("Motor gücü:", enginePower)
    print("Çekiş:", drivetrain)
    print("Değişen parça:", replacedParts)
    print("Takasa uygunluk:", exchange)
    print("Kimden:", fromWhom)

    newCar = [
        int(dict_Marka.get(brand)),
        int(dict_Seri.get(series)),
        int(dict_Model.get(model)),
        productionYear,
        mileage,
        int(dict_VitesTipi.get(gearbox)),
        int(dict_YakıtTipi.get(fuelType)),
        int(dict_KasaTipi.get(bodyType)),
        enginePower,
        int(dict_Cekis.get(drivetrain)),
        int(dict_BoyaDegisen.get(replacedParts)),
        int(dict_TakasaUygun.get(exchange)),
        int(dict_Kimden.get(fromWhom))
    ]
    newCar = np.array(newCar)
    print("New Car \n", newCar)

    y_pred = xgb.predict(newCar.reshape((1, -1)))
    print("y pred:", y_pred)

    st.write("Arabanın Fiyatı: ", str(y_pred[0])+ ' TL')
