# learn-machine-learning

## 機械学習において注意すること
* 入力変数通しが高い相関関係を持つと思うような学習結果を得られない可能性がある

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

# パラメータの分布をヒストグラムで可視化
plt.figure(figsize=(10, 7))
plt.bar(x=columns, height=model.coef_);

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
