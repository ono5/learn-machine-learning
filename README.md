# learn-machine-learning

## 機械学習において注意すること
* 入力変数通しが高い相関関係を持つと思うような学習結果を得られない可能性がある
  - 対処方法として、Partial Least Squares (PLS)が有効
  - データには欠損値や外れ値、Typeの違いなどあるため、データの前処理が必要

## 教師あり学習の目的
教師あり学習の目的は、未知なデータに対しても高い性能を発揮するようにモデルを学習すること。
学習時に用いなかったデータに対しては予測値と目標値の差異が大きくなってしまう現象を、過学習 (overfitting)と呼ぶ。

## Sklearnのデータセット読み込み

```
# データセットの読み込み
from sklearn.datasets import load_boston
dataset = load_boston()
x, t = dataset.data, dataset.target
columns = dataset.feature_names
```

## PandasのDataFrameに変更
```
df = pd.DataFrame(x, columns=columns)
```

## PandasのDataFrameに列を追加

```
df['Target'] = t
```

## 入力変数と出力変数の切り分け

```
# 入力変数と出力変数の切り分け
t = df['Target'].values
x = df.drop(labels=['Target'], axis=1).values
```

## .valuesに意味
pandas.core.series.Series -> numpy.ndarrayに変換
```
type(df['Target']), type(df['Target'].values)
```

scikit-learn を用いて機械学習アルゴリズムを実装を行う際には NumPy の ndarray に変換する必要がある。

## データセットの分割

* 実力をつけるための勉強に使うデータの集まりを、学習用データセット (training dataset)
* 実力をはかるために使うデータの集まりを、テスト用データセット (test dataset)
* ホールドアウト法 (holdout method) --- 単純に全体の何割かを学習用データセットとし、残りをテスト用データセットとする、といった分割を行う方法

```
from sklearn.model_selection import train_test_split

# 2 つに分割
x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.3, random_state=0)
```

## モデルの学習・検証
### 1. Step 1：モデルの定義
モデルの定義ではどの機械学習アルゴリズムを使用してモデルの構築を行うのかを定義。

```
# 重回帰分析
from sklearn.linear_model import LinearRegression
model = LinearRegression()
```

### 2. Step 2：モデルの学習
model を用いて学習を実行するには、fit() の引数に入力値 x と目標値 t を与える。

```
# Step 2：モデルの学習
model.fit(x_train, t_train)
```

### 3. Step 3：モデルの検証
モデルの学習が完了したら精度の検証を行う。

LinearRegressionクラスは score() メソッドを提供しており、入力変数と出力変数を与えると
学習済みのモデルを用いて計算した決定係数 (coefficient of determination) という指標を返す。

決定係数の最大値は 1 であり、値が大きいほど（1 に近いほど）モデルが与えられたデータに当てはまっていることを表す。

```
print('train score : ', model.score(x_train, t_train))
print('test score : ', model.score(x_test, t_test))
```

### 推論
学習が終わったモデルに、新たな入力値を与えて予測値を計算させる。

```
# 推論
y = model.predict(x_test)

print('予測値: ', y[0])
print('目標値: ', t_test[0])
```

### 重回帰分析のパラメータ
```
# 学習後のパラメータ w
model.coef_

# 学習後のバイアス b
model.intercept_
```

## 過学習を抑える方法

過学習を抑制するアプローチ。

1. データセットのサンプルサイズを増やす
2. ハイパーパラメータを調整する
3. 他のアルゴリズムを使用する

## 可視化

```
# 箱を準備
fig = plt.figure(figsize=(7, 10))

# 重回帰分析
ax1 = fig.add_subplot(2, 1, 1)
ax1.bar(x=columns, height=model.coef_)
ax1.set_title('Linear Regression')

# リッジ回帰
ax2 = fig.add_subplot(2, 1, 2)
ax2.bar(x=columns, height=ridge.coef_)
ax2.set_title('Ridge Regression');
```

## 多重共線性
入力変数同士の相関が強いものが含まれている場合、多重共線性 (Multicollinearity) という問題が起こる。

* 対処方法
  - データセットの中から相関の高い入力変数のどちらかを削除する
  - 使用するアルゴリズムを変更する

`.corr()` を使用して相関関係を表す数値（相関係数）を確認することができる。

相関係数は 1 が最大で、その 2 つの変数が完全に正の相関があることを表す。

```
# 相関係数の算出
df_corr = df.corr()

# 可視化
plt.figure(figsize=(12, 8))
sns.heatmap(df_corr.iloc[:20, :20], annot=True);

# 特定の変数を可視化
sns.jointplot(x='x1', y='x16', data=df);
```

### 多重共線性の対処方法 -> Partial Least Squares (PLS)
Partial Least Squares (PLS) は多重共線性の問題の対処法として有力なアルゴリズム。

* ステップ1: 入力値と目標値の共分散が最大になるように主成分を抽出
* ステップ2: 抽出された主成分に対して重回帰分析を用いてモデルの学習を行う

「主成分を抽出」とは、100 個の特徴量があった場合にその特徴量の数を分散を用いて 10 や 20 個などに削減する手法（次元削減）。

主成分を抽出するときに目標値の情報を使う。

```
# モデルの定義（ n_components:7 とする）
# n_componentsは何次元に落とすかのパラメータであり、試行錯誤しながらこの数値をいじる
from sklearn.cross_decomposition import PLSRegression
pls = PLSRegression(n_components=7)

# モデルの学習
pls.fit(x_train, t_train)

# モデルの検証
print('train score : ', pls.score(x_train, t_train))
print('test score : ', pls.score(x_test, t_test))
```

## データの前処理
### 重複行の確認 / 削除

```
# 重複行の確認

# keep=False で重複データをすべて True , それ以外を False
df.duplicated(keep=False)

# 要約を出力
df.duplicated(keep=False).value_counts()

# 重複行の削除
df.drop_duplicates()
```

### 欠損値処理
csv中の空白は`セル欠損値(NaN)` となる。


|方法|使う場面|
|:--|:--|
欠損値を含む行（サンプル）を取り除く|欠損値の数が同一列内で多くない場合
欠損値を含む列（入力変数）を取り除く|欠損値の数が同一列内で多すぎる場合

```
# 欠損値の確認
df.isnull()[:5] # [:5] で表示する行を指定

# 欠損値の数を確認
df.isnull().sum()

# 欠損値の除去(dropna(subset=['列名'], axis=0))
# 列方向に削除 -> axisを１
# dropna()の場合は欠損値が含まれる行が全て削除

# 行の削除
df = df.dropna(subset=['charges'])
# 複数指定できる
df = df.dropna(subset=['price','horsepower','peak-rpm'])

# 列ごと削除
df = df.drop(labels='rank', axis=1)
```

### 欠損値の補完
欠損値補完は、欠損値を他の値で埋める方法。

以下の方法で埋める。
1. 平均値(一般的に用いられる)
2. 中央値

#### 数値
`plt.hist(df['bmi'])`などで、データのばらつき、外れ値などを事前に確認する。

```
# データのばらつき確認
plt.hist(df['bmi'])

# 平均値の確認
df['bmi'].mean()

# 欠損値を平均値で補完
# fillna({'補完対象の列名'：補完する値})
# df['列名'].mean() で指定した列の平均値を算出
df = df.fillna({'bmi':df['bmi'].mean()})
```

#### 文字列
文字列に対しての欠損値補完では 最頻値 を使用するケースが多い。

```
# 入力文字の種類を確認(array(['yes', nan, 'no'], dtype=object))
df['smoker'].unique()

# 一番多い文字数を確認
df['smoker'].mode()

0    no
dtype: object

# noを取り出す
df['smoker'].mode()[0]

# 最頻値を使用して欠損値を補完(noが一番多いので、noで埋める)
df = df.fillna({'smoker':df['smoker'].mode()[0]})
```

## 特徴量変換

### カテゴリカル変数の取り扱い
文字列データ含め、カテゴリを表すデータをカテゴリカル変数(質的変数)と呼ぶ。

「男性, 女性」というような文字列データはカテゴリカル変数であり、
男性を0,女性を1と置き換えた「0, 1」のデータもカテゴリカル変数。

```
# カテゴリカル変数を含んだデータのみを抽出(文字列データを取り出し)
df_obj = df.select_dtypes(include='object')

# ユニークな値の数を確認
df_uni = df_obj.nunique()
```

### Label Encoding　
Label Encoding は各カテゴリに 0, 1, 2, ... と数値を割り振る変換。

```
# モデルの宣言
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

le.fit(df['gender'])

# 適用
le.transform(df['gender'])

# 値の入れ替え
df['gender'] = le.transform(df['gender'])

# ラベル変換された値についての確認
le.classes_
```

小技。

```
from sklearn.preprocessing import LabelEncoder
# ユニークな値が 2 種類の列名のみ取得
df_cols = df_uni[df_uni == 2].index

for col in df_cols:
  le = LabelEncoder()
  le.fit(df[col])
  df[col] = le.transform(df[col])
```

### One-Hot Encoding
One-Hot Encoding はダミー変数化とも言われ、各カテゴリの値ごとに 0, 1の列を作成する。

```
# drop_firstは、Trueにすることで変換後の先頭列を除去
df = pd.get_dummies(df, drop_first=True)
```

入力変数の数が増える点がOne-Hot Encodingのデメリット。

## データの統計量

```
# データ統計量の確認
df.describe()
```

## 可視化

```
# ヒストグラム表示
plt.figure(figsize=(10, 7))
plt.bar(x=columns, height=model.coef_);

plt.hist(df['bmi'])
```

## 特徴量エンジニアリング
特徴量エンジニアリングは精度をあげる為の前処理の１つで、現在あるデータを用い、より有用なデータを作成すること。

多いカテゴリをより有用な少ないカテゴリに分け直す。

車を例に上げる。

```
# クラス分けのリストの定義
class_3 = ['audi', 'bmw', 'jaguar', 'mercedes-benz', 'porsche']
class_2 = ['alfa-romero', 'chevrolet',  'mercury', 'volvo', 'toyota', 'plymouth', 'dodge']
class_1 = ['honda', 'isuzu', 'mazda', 'mitsubishi', 'nissan', 'peugot', 'saab', 'subaru', 'volkswagen']

# それぞれを置換するリストの作成
maker_class = []
for i in df_obj['make']:
    if i in class_3:
        maker_class.append(3)
    elif i in class_2:
        maker_class.append(2)
    elif i in class_1:
        maker_class.append(1)

# リストの確認
maker_class[:10]
[2, 2, 2, 3, 3, 3, 3, 3, 3, 3]
```

## 外れ値除去
外れ値を除去することで、精度向上を目指す。

### 3σ 法（平均値ベース）
平均値をベースに外れ値を除去する。

```
mu = df['price'].mean() # 平均値
sigma = df['price'].std() # 標準偏差
# 3σ法の中身を取得
df3 = df[(mu - 3 * sigma <= df['price']) & (df['price'] <= mu + 3 * sigma)]
```

### ハンペル判別法（中央値ベース）
中央値ベースに外れ値を除去する。

```
# MADを算出
median = df['price'].median()

# absは指定の値を絶対値に変換。np.medianで中央値を算出
# 1.4826は公式の数値
MAD = 1.4826 * np.median(abs(df['price']-median))
```
