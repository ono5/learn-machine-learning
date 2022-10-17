# Ridge回帰とLasso回帰
Ridge回帰とLasso回帰は過学習を抑えるために正則化項を追加した線形回帰。

過学習を抑制するアプローチとして有効。

![スクリーンショット 2022-10-16 5 43 21](https://user-images.githubusercontent.com/20691160/196006824-cc14a123-c002-440f-8dae-e1143b136767.png)

正則化項は、二乗誤差和に追加された項でパラメータの変動を抑える(罰則を与える)。
そもそも過学習とは、学習データに過剰に適合してしまい全く使えない予測モデルを作ってしまうことを指す。

## Ridge 回帰（リッジ回帰）
Redge回帰は、重回帰分析に対して重みの 2 乗で表現される L2 ノルムを用いて正則化を行うことで、モデルの過度な複雑さに罰則を課して過学習を抑制する手法。

正則化項の α が罰則の強さを表し、この値を大きくするほど小さな幅でパラメータが調整される。

```
# モデルの定義、ハイパーパラメータの値を設定
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1)

# モデルの学習
ridge.fit(x_train, t_train)

# モデルの検証
print('train score : ', ridge.score(x_train, t_train))
print('test score : ', ridge.score(x_test, t_test))

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

## Lasso 回帰（ラッソ回帰）
Lasso 回帰は重回帰分析に対して、||w||1 で表現される L1 ノルムを使用して正則化を行う手法。
L2ノルムとは違い、重みの絶対値を求める。

Lasso回帰の特徴として不要な入力変数を特定し、該当する重み w を 0 にする事で実質的に入力変数の種類を減らすことができる。

このとき生成される重み w のベクトル W は 0 を多く含み、これをスパース性があると表現する。

入力変数が多すぎるときパラメータを0にして絞る。

```
# モデルの定義
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=1)

# モデルの学習
lasso.fit(x_train, t_train)

# モデルの検証
print('train score : ', lasso.score(x_train, t_train))
print('test score : ', lasso.score(x_test, t_test))

# 0 になっていない特徴量の数
print('元の特徴量の数 : ', x.shape[1])
print('Lasso の特徴量 : ', np.sum(lasso.coef_ != 0))

# アルファを変更
lasso_005 = Lasso(alpha=0.05)
lasso_005.fit(x_train, t_train)

print('train score : ', lasso_005.score(x_train, t_train))
print('test score : ', lasso_005.score(x_test, t_test))

# 0 になっていない特徴量の数
print('元の特徴量の数 : ', x.shape[1])
print('Lasso005 の特徴量 : ', np.sum(lasso_005.coef_ != 0))

fig = plt.figure(figsize=(7, 15))

# 重回帰分析
ax1 = fig.add_subplot(3, 1, 1)
ax1.bar(x=columns, height=model.coef_)
ax1.set_title('Linear Regression')

# lasso
ax2 = fig.add_subplot(3, 1, 2)
ax2.bar(x=columns, height=lasso.coef_)
ax2.set_title('Lasso Regression 1')

# lasso_005
ax3 = fig.add_subplot(3, 1, 3)
ax3.bar(x=columns, height=lasso_005.coef_)
ax3.set_title('Lasso Regression 0.05');
```
