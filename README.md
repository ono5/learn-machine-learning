# learn-machine-learning

## Sklearnのデータセット読み込み

```
# データセットの読み込み
from sklearn.datasets import load_boston
dataset = load_boston()
x, t = dataset.data, dataset.target
columns = dataset.feature_names
```

## Pandas の DataFrameに変更
```
df = pd.DataFrame(x, columns=columns)
```
