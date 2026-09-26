import pandas as pd, numpy as np
df = pd.read_csv('diabetes.csv')

# 1) 정답(Outcome) 기반으로 채워진 대체값 -> 결측(NaN)으로 복원
df.loc[df.Insulin.isin([102.5, 169.5]), 'Insulin'] = np.nan
df.loc[df.SkinThickness.isin([27, 32]), 'SkinThickness'] = np.nan
df['Insulin_missing'] = df.Insulin.isna().astype(int)
df['Skin_missing'] = df.SkinThickness.isna().astype(int)

# 2) 결측은 Outcome과 무관한 전체 중앙값으로 대체
feat = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
df[feat] = df[feat].fillna(df[feat].median())

# 3) 이상치는 1.5*IQR 경계로 clip
q1, q3 = df[feat].quantile(.25), df[feat].quantile(.75)
df[feat] = df[feat].clip(q1 - 1.5*(q3-q1), q3 + 1.5*(q3-q1), axis=1)

df.to_csv('diabetes_preprocessed.csv', index=False)
print(df.shape, df.isna().sum().sum(), 'NaN'); print(df.describe().T[['min','50%','max']])
